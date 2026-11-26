//! benilla's own: a relaunch that rejoins the world. The Graphics API prompt's "Yes" restarts the
//! client ([`crate::video::relaunch_if_requested`]); with a character in the world, the old
//! process hands the new one a [`ResumeTicket`] and the new one dials the world server with the
//! session key it carries, skipping the login and character screens. vmangos keeps a dropped
//! character in the world for two minutes (`ForceLogoutDelay`, `WorldSession.cpp`) and logs the
//! next login of that character into the same `Player` (`alreadyOnline`, `CharacterHandler.cpp`),
//! so to everyone else it never left.
//!
//! The ticket travels only through an anonymous pipe on the new process's stdin, which no other
//! process can open: never a file, the command line or the environment. It holds no password;
//! the session key it holds is the one realmd stored at the last logon, which the world server
//! accepts only from the address that logged in (`last_ip`) and only the same day
//! (`WorldSocket.cpp`), and which the next real logon replaces. Any failure (no ticket, a bad one,
//! a refused key) lands on the login screen, as a plain launch does.

use std::io::{Read, Write};
use std::sync::Mutex;
use std::time::Duration;

use bevy::prelude::*;

/// The marker argument a resumed process is started with; only with it is stdin read at all.
pub(crate) const RESUME_ARG: &str = "--benilla-resume";

/// The ticket format's first line.
const MAGIC: &str = "benilla-resume 1";

/// How long a resumed process waits for its ticket before starting as a plain launch: the parent
/// writes it the moment the child exists, so this only bounds a launch made by hand.
const READ_TIMEOUT: Duration = Duration::from_secs(2);

/// A ticket is a few hundred bytes; more is not one.
const MAX_TICKET_BYTES: u64 = 4096;

/// What a new process needs to rejoin the world as this one is in it. No `Debug`: it holds the
/// session key, which must never reach a log.
#[derive(Clone)]
pub(crate) struct ResumeTicket {
    /// The account name the session key was proved for.
    pub(crate) user: String,
    /// The realmlist the logon dialled, kept for the new session's request.
    pub(crate) host: String,
    /// The locale tag the logon carried.
    pub(crate) locale: u32,
    /// The SRP6 session key realmd stored for this account at the last logon.
    pub(crate) session_key: Vec<u8>,
    /// The realm the world session is on.
    pub(crate) realm: benilla_protocol::RealmInfo,
    /// The character in the world.
    pub(crate) guid: u64,
}

impl Drop for ResumeTicket {
    /// Best effort: the key's bytes are overwritten before the allocation is freed.
    fn drop(&mut self) {
        wipe(&mut self.session_key);
    }
}

/// Overwrite `bytes` with zeros in a way the optimiser keeps.
fn wipe(bytes: &mut [u8]) {
    for b in bytes.iter_mut() {
        // SAFETY: `b` is a valid, aligned `&mut u8` for the duration of the write.
        unsafe { std::ptr::write_volatile(b, 0) };
    }
}

/// Lock a slot whatever an earlier panic left in it: a ticket is plain data.
fn lock<T>(slot: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

// ── The running process's side ─────────────────────────────────────────────────────────────────

/// The ticket for the world session standing now, posted by the IO thread at the world entry and
/// cleared when that session ends, so a relaunch never carries a stale one.
static LIVE: Mutex<Option<ResumeTicket>> = Mutex::new(None);

/// The IO thread entered the world: this is what a relaunch would carry.
pub(crate) fn post(ticket: ResumeTicket) {
    *lock(&LIVE) = Some(ticket);
}

/// The world session ended (a logout, a refusal, a dropped link): nothing to resume.
pub(crate) fn clear() {
    *lock(&LIVE) = None;
}

/// The ticket a relaunch will hand over, moved out of [`LIVE`] when the relaunch is asked for,
/// while the session still stands: the exit that follows ends the IO thread's stream, whose
/// [`clear`] could otherwise beat [`spawn`] to it.
static ARMED: Mutex<Option<ResumeTicket>> = Mutex::new(None);

/// The relaunch was asked for: keep the standing session's ticket for [`spawn`].
pub(crate) fn arm() {
    let live = lock(&LIVE).clone();
    *lock(&ARMED) = live;
}

/// Start `command` as the relaunched client: with a world session armed ([`arm`]), as a resumed
/// one, its ticket written into the child's stdin and the pipe closed; otherwise as a plain
/// launch. A failed handover still leaves the child running, which then starts at the login
/// screen.
pub(crate) fn spawn(mut command: std::process::Command) -> std::io::Result<()> {
    let ticket = lock(&ARMED).take();
    let Some(ticket) = ticket else {
        command.spawn()?;
        return Ok(());
    };
    let Some(mut text) = encode(&ticket) else {
        warn!("resume: the session cannot be written into a ticket — relaunching to the login screen");
        command.spawn()?;
        return Ok(());
    };
    command.arg(RESUME_ARG).stdin(std::process::Stdio::piped());
    let spawned = command.spawn();
    let written = match spawned {
        Ok(mut child) => match child.stdin.take() {
            Some(mut pipe) => pipe.write_all(text.as_bytes()).and_then(|()| pipe.flush()),
            None => Ok(()),
        },
        Err(e) => {
            wipe_string(&mut text);
            return Err(e);
        }
    };
    wipe_string(&mut text);
    if let Err(e) = written {
        warn!("resume: handing the session over failed ({e}) — the new client starts at login");
    } else {
        info!("resume: relaunched with the session handed over");
    }
    Ok(())
}

// ── The resumed process's side ─────────────────────────────────────────────────────────────────

/// The ticket this process was started with, until the IO thread takes it.
static INCOMING: Mutex<Option<ResumeTicket>> = Mutex::new(None);

/// The character and realm of [`INCOMING`], for the app's roster to seed itself from without
/// touching the key.
static INCOMING_PICK: Mutex<Option<(u64, benilla_protocol::RealmInfo)>> = Mutex::new(None);

/// What reading the ticket came to, logged once logging is up ([`take_incoming`]).
static READ_NOTE: Mutex<Option<String>> = Mutex::new(None);

/// At launch, before the IO thread exists: when started with [`RESUME_ARG`], read the ticket the
/// parent writes into stdin, waiting at most [`READ_TIMEOUT`]. Without the marker stdin is never
/// touched.
pub(crate) fn read_from_parent() {
    if !std::env::args_os().skip(1).any(|a| a == RESUME_ARG) {
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    let reader = std::thread::Builder::new()
        .name("resume-read".into())
        .spawn(move || {
            let mut bytes = Vec::new();
            let read = std::io::stdin()
                .lock()
                .take(MAX_TICKET_BYTES)
                .read_to_end(&mut bytes);
            let _ = tx.send(read.map(|_| bytes));
        });
    let note = if reader.is_err() {
        "the ticket reader could not start".to_string()
    } else {
        match rx.recv_timeout(READ_TIMEOUT) {
            Ok(Ok(mut bytes)) => {
                let ticket = std::str::from_utf8(&bytes).ok().and_then(decode);
                wipe(&mut bytes);
                match ticket {
                    Some(ticket) => {
                        *lock(&INCOMING_PICK) = Some((ticket.guid, ticket.realm.clone()));
                        let note = format!(
                            "ticket for {} on {} read — rejoining the world",
                            ticket.user, ticket.realm.name
                        );
                        *lock(&INCOMING) = Some(ticket);
                        note
                    }
                    None => "the ticket was not a valid one".to_string(),
                }
            }
            Ok(Err(e)) => format!("reading the ticket failed ({e})"),
            Err(_) => "no ticket arrived in time".to_string(),
        }
    };
    *lock(&READ_NOTE) = Some(note);
}

/// The IO thread's first cycle: the ticket, once. Logs how the read went.
pub(crate) fn take_incoming() -> Option<ResumeTicket> {
    if let Some(note) = lock(&READ_NOTE).take() {
        info!("resume: {note}");
    }
    lock(&INCOMING).take()
}

/// The character and realm a resumed process re-enters as, once, for the roster to seed.
fn take_incoming_pick() -> Option<(u64, benilla_protocol::RealmInfo)> {
    lock(&INCOMING_PICK).take()
}

// ── The ticket's text ──────────────────────────────────────────────────────────────────────────

/// One `key=value` line per field after [`MAGIC`]. `None` if a text field could break the format.
fn encode(t: &ResumeTicket) -> Option<String> {
    let breaks = |s: &str| s.contains('\n') || s.contains('\r');
    if [&t.user, &t.host, &t.realm.name, &t.realm.address]
        .iter()
        .any(|s| breaks(s))
    {
        return None;
    }
    if t.user.is_empty() || t.realm.name.is_empty() || t.realm.address.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(512);
    out.push_str(MAGIC);
    out.push('\n');
    let mut line = |k: &str, v: &str| {
        out.push_str(k);
        out.push('=');
        out.push_str(v);
        out.push('\n');
    };
    line("user", &t.user);
    line("host", &t.host);
    line("locale", &t.locale.to_string());
    line("key", &to_hex(&t.session_key));
    line("guid", &t.guid.to_string());
    line("realm.name", &t.realm.name);
    line("realm.address", &t.realm.address);
    line("realm.population", &t.realm.population.to_string());
    line("realm.characters", &t.realm.characters.to_string());
    line("realm.type", &t.realm.realm_type.to_string());
    line("realm.flags", &t.realm.flags.to_string());
    line("realm.category", &t.realm.category.to_string());
    line("realm.id", &t.realm.id.to_string());
    Some(out)
}

/// The reverse of [`encode`]; `None` for anything short of a complete, well-formed ticket.
fn decode(text: &str) -> Option<ResumeTicket> {
    let mut lines = text.lines();
    if lines.next()? != MAGIC {
        return None;
    }
    let mut fields = std::collections::HashMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (k, v) = line.split_once('=')?;
        fields.insert(k, v);
    }
    let get = |k: &str| fields.get(k).copied();
    let num = |k: &str| get(k).and_then(|v| v.parse::<u64>().ok());
    let session_key = from_hex(get("key")?)?;
    if session_key.is_empty() {
        return None;
    }
    let user = get("user")?.to_string();
    let realm_name = get("realm.name")?.to_string();
    let address = get("realm.address")?.to_string();
    if user.is_empty() || realm_name.is_empty() || address.is_empty() {
        return None;
    }
    Some(ResumeTicket {
        user,
        host: get("host")?.to_string(),
        locale: u32::try_from(num("locale")?).ok()?,
        session_key,
        realm: benilla_protocol::RealmInfo {
            name: realm_name,
            address,
            population: get("realm.population")?.parse().ok()?,
            characters: u8::try_from(num("realm.characters")?).ok()?,
            realm_type: u32::try_from(num("realm.type")?).ok()?,
            flags: u8::try_from(num("realm.flags")?).ok()?,
            category: u8::try_from(num("realm.category")?).ok()?,
            id: u8::try_from(num("realm.id")?).ok()?,
        },
        guid: num("guid")?,
    })
}

fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(DIGITS[usize::from(b >> 4)] as char);
        s.push(DIGITS[usize::from(b & 0x0f)] as char);
    }
    s
}

fn from_hex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let digit = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    s.as_bytes()
        .chunks(2)
        .map(|pair| Some(digit(pair[0])? << 4 | digit(pair[1])?))
        .collect()
}

/// Overwrite a string's bytes before it is freed.
fn wipe_string(s: &mut String) {
    let mut bytes = std::mem::take(s).into_bytes();
    wipe(&mut bytes);
}

// ── The app's side of a resumed launch ─────────────────────────────────────────────────────────

/// A resumed launch's pick is armed in the roster and the world not yet entered: a refusal
/// before the entry must not leave it to auto-enter whatever the player logs into next.
#[derive(Resource, Default)]
struct ResumeSeed(bool);

/// Seeds the roster with a resumed launch's character, so the existing reconnect answer
/// ([`crate::glue_session`]) asks the roster and enters the world with it.
pub(crate) struct ResumePlugin;

impl Plugin for ResumePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ResumeSeed>()
            .add_systems(Startup, seed_roster)
            // After the roster policy has read the frame's roster with the pick still armed.
            .add_systems(
                Update,
                settle_seed.after(crate::glue_session::edges::SessionEdges),
            );
    }
}

/// The kept pick and realm, as an unattended reconnect keeps them.
fn seed_roster(
    roster: Option<ResMut<crate::glue_session::roster::Roster>>,
    mut seed: ResMut<ResumeSeed>,
) {
    let Some(mut roster) = roster else {
        return;
    };
    if let Some((guid, realm)) = take_incoming_pick() {
        roster.pending_pick = Some(guid);
        roster.realm = Some(realm);
        seed.0 = true;
    }
}

/// The world entry ends the seed; a failure before it drops the pick it armed.
fn settle_seed(
    mut entered: MessageReader<crate::net::EnteredWorldMessage>,
    mut failed: MessageReader<crate::net::LoginFailedMessage>,
    roster: Option<ResMut<crate::glue_session::roster::Roster>>,
    mut seed: ResMut<ResumeSeed>,
) {
    let entered = entered.read().count() != 0;
    let failed = failed.read().count() != 0;
    if !seed.0 {
        return;
    }
    if entered {
        seed.0 = false;
    } else if failed {
        info!("resume: the world refused the handed-over session — log in as usual");
        if let Some(mut roster) = roster {
            roster.pending_pick = None;
        }
        seed.0 = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ticket() -> ResumeTicket {
        ResumeTicket {
            user: "PLAYER".into(),
            host: "127.0.0.1".into(),
            locale: 0x656e5553,
            session_key: (0..40).collect(),
            realm: benilla_protocol::RealmInfo {
                name: "My Realm".into(),
                address: "127.0.0.1:8085".into(),
                population: 1.5,
                characters: 3,
                realm_type: 1,
                flags: 0x20,
                category: 1,
                id: 1,
            },
            guid: 42,
        }
    }

    #[test]
    fn a_ticket_survives_its_text() {
        let t = ticket();
        let back = decode(&encode(&t).expect("encodes")).expect("decodes");
        assert_eq!(back.user, t.user);
        assert_eq!(back.host, t.host);
        assert_eq!(back.locale, t.locale);
        assert_eq!(back.session_key, t.session_key);
        assert_eq!(back.guid, t.guid);
        assert_eq!(back.realm.name, t.realm.name);
        assert_eq!(back.realm.address, t.realm.address);
        assert_eq!(back.realm.population, t.realm.population);
        assert_eq!(back.realm.id, t.realm.id);
    }

    #[test]
    fn anything_short_of_a_ticket_is_none() {
        assert!(decode("").is_none());
        assert!(decode("not a ticket\nuser=x").is_none());
        let text = encode(&ticket()).unwrap();
        let without_key: String = text
            .lines()
            .filter(|l| !l.starts_with("key="))
            .map(|l| format!("{l}\n"))
            .collect();
        assert!(decode(&without_key).is_none());
        assert!(decode(&text.replace("key=", "key=zz")).is_none());
    }

    #[test]
    fn a_name_that_would_break_the_format_writes_no_ticket() {
        let mut t = ticket();
        t.realm.name = "two\nlines".into();
        assert!(encode(&t).is_none());
    }
}
