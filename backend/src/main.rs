//! Press-to-Handstand self-evolving coach.
//!
//! A Rust engine (std-only) that runs a genetic algorithm evolving a small
//! neural-network "coach", and a minimal HTTP server that drives a TypeScript
//! mobile web client. The coach turns an athlete's current capacity profile
//! into an optimal training focus that, under the built-in progression
//! simulator, reaches press-to-handstand fastest and safest.

use p2h_engine::{evo, model};

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const ADDR: &str = "0.0.0.0:8080";
/// Weeks shown in the recommended pathway.
const PLAN_WEEKS: usize = model::WEEKS;

fn main() {
    let evolution = Arc::new(Mutex::new(evo::Evolution::new(1337)));
    let running = Arc::new(AtomicBool::new(true));

    // Background evolution loop.
    {
        let evolution = evolution.clone();
        let running = running.clone();
        std::thread::spawn(move || loop {
            if running.load(Ordering::Relaxed) {
                evolution.lock().unwrap().step();
                std::thread::sleep(std::time::Duration::from_millis(2));
            } else {
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
        });
    }

    let listener = TcpListener::bind(ADDR).expect("bind");
    println!("[p2h] listening on http://{ADDR}");
    let _ = std::io::stdout().flush();

    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                let evolution = evolution.clone();
                let running = running.clone();
                std::thread::spawn(move || {
                    let _ = handle(s, evolution, running);
                });
            }
            Err(_) => continue,
        }
    }
}

// ---------------------------------------------------------------------------
// HTTP request parsing
// ---------------------------------------------------------------------------

struct Request {
    method: String,
    path: String,
    body: String,
}

fn read_request(stream: &mut TcpStream) -> Option<Request> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    // Read headers.
    loop {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if buf.len() > 1_000_000 {
            return None;
        }
    }
    let head_end = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| p + 4)
        .unwrap_or(buf.len());
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let req_line = lines.next().unwrap_or("");
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    // Content length.
    let mut content_length = 0usize;
    for line in lines {
        let lower = line.to_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        }
    }

    // Read remaining body bytes.
    let body_start = head_end;
    let mut body_bytes = buf[body_start..].to_vec();
    while body_bytes.len() < content_length {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            break;
        }
        body_bytes.extend_from_slice(&tmp[..n]);
    }
    let body = String::from_utf8_lossy(&body_bytes[..content_length.min(body_bytes.len())]).to_string();

    Some(Request {
        method,
        path,
        body,
    })
}

fn write_response(stream: &mut TcpStream, status: &str, content_type: &str, body: &str) {
    let resp = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Methods: GET,POST,OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(resp.as_bytes());
}

fn respond_json(stream: &mut TcpStream, body: &str) {
    write_response(stream, "200 OK", "application/json; charset=utf-8", body);
}

// ---------------------------------------------------------------------------
// Static file serving
// ---------------------------------------------------------------------------

fn serve_static(stream: &mut TcpStream, path: &str) {
    let rel = path.trim_start_matches('/');
    let public = "public";
    let full = if rel.is_empty() || rel == "index.html" {
        format!("{public}/index.html")
    } else {
        format!("{public}/{rel}")
    };
    // Path protection.
    if full.contains("..") {
        write_response(stream, "404 Not Found", "text/plain", "not found");
        return;
    }
    match std::fs::read(&full) {
        Ok(bytes) => {
            let s = String::from_utf8_lossy(&bytes).to_string();
            let ct = match rel.rsplit('.').next() {
                Some("js") => "application/javascript; charset=utf-8",
                Some("css") => "text/css; charset=utf-8",
                Some("svg") => "image/svg+xml",
                Some("png") => "image/png",
                Some("json") => "application/json",
                _ => "text/html; charset=utf-8",
            };
            write_response(stream, "200 OK", ct, &s);
        }
        Err(_) => write_response(stream, "404 Not Found", "text/plain", "not found"),
    }
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn json_esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            _ => o.push(c),
        }
    }
    o
}

fn floats_json(v: &[f32]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x:.4}")).collect();
    format!("[{}]", items.join(","))
}

// ---------------------------------------------------------------------------
// API handlers
// ---------------------------------------------------------------------------

fn api_evolution(evolution: &Arc<Mutex<evo::Evolution>>, running: &AtomicBool) -> String {
    let ev = evolution.lock().unwrap();
    let hist = &ev.history;
    // Downsample the history for transport if it's very long.
    let max_points = 300;
    let step = (hist.len() / max_points).max(1);
    let sampled: Vec<f32> = hist.iter().step_by(step).copied().collect();
    format!(
        r#"{{"gen":{},"best_fitness":{:.3},"population":{},"profiles":{},"running":{},"history":{}}}"#,
        ev.gen,
        ev.best_fitness,
        ev.population.len(),
        ev.profiles.len(),
        running.load(Ordering::Relaxed),
        floats_json(&sampled),
    )
}

fn api_meta() -> String {
    let caps: Vec<String> = model::CAPACITY_NAMES
        .iter()
        .zip(model::CAPACITY_HINT.iter())
        .map(|(n, h)| {
            format!(r#"{{"name":"{}","hint":"{}"}}"#, json_esc(n), json_esc(h))
        })
        .collect();
    let skills: Vec<String> = model::SKILL_NAMES
        .iter()
        .map(|n| format!(r#"{{"name":"{}"}}"#, json_esc(n)))
        .collect();
    let cur = model::SKILL_COUNT.saturating_sub(1);
    // mark the final skill as the goal
    format!(
        r#"{{"capacities":[{c}],"skills":[{s}],"weeks":{w},"threshold":{t},"goal":{g},"goalName":"{goalName}"}}"#,
        c = caps.join(","),
        s = skills.join(","),
        w = model::WEEKS,
        t = evo::PRESS_THRESHOLD,
        g = cur,
        goalName = json_esc(model::SKILL_NAMES[model::SKILL_COUNT - 1]),
    )
}

fn api_recommend(evolution: &Arc<Mutex<evo::Evolution>>, body: &str) -> String {
    // Parse {"caps":[...8 floats]}
    let (_, caps_raw) = body
        .split_once("\"caps\":")
        .unwrap_or(("", &body[body.len().min(0)..]));
    let mut caps = [0.5f32; model::N_CAP];
    if let Some(open) = caps_raw.find('[') {
        if let Some(close) = caps_raw[open + 1..].find(']') {
            let inner = &caps_raw[open + 1..open + 1 + close];
            for (i, tok) in inner.split(',').enumerate() {
                if i >= model::N_CAP {
                    break;
                }
                if let Ok(v) = tok.trim().parse::<f32>() {
                    caps[i] = v.clamp(0.0, 1.0);
                }
            }
        }
    }

    // Two candidate coaches:
    //   * the GA-evolved neural network (self-evolving),
    //   * the deterministic greedy baseline (a reliable safety net).
    // We simulate both and return whichever attains the higher press score.
    let ev = evolution.lock().unwrap();
    let net = ev.best_net();
    let evolved = model::simulate(
        &caps,
        |c| {
            let logits = net.forward(c);
            model::softmax(&logits)
        },
        PLAN_WEEKS,
        true,
    );
    let baseline = model::simulate(&caps, |c| model::greedy_focus(c), PLAN_WEEKS, true);

    // Prefer the one reaching the press earlier / with higher attainment.
    let evo_score = evolved.final_score;
    let base_score = baseline.final_score;
    let (result, source) = if evo_score >= base_score {
        (evolved, "evolved")
    } else {
        (baseline, "baseline")
    };

    // Group consecutive weeks into blocks by dominant skill.
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new(); // (skill, start_week, count)
    let mut idx = 0usize;
    while idx < result.plan.len() {
        let skill = result.plan[idx];
        let start = idx;
        let mut end = idx;
        while end + 1 < result.plan.len() && result.plan[end + 1] == skill {
            end += 1;
        }
        blocks.push((skill, start, end - start + 1));
        idx = end + 1;
    }

    let mut focus_sum = vec![0.0f32; model::SKILL_COUNT];
    for (skill, _, count) in &blocks {
        focus_sum[*skill] += *count as f32;
    }
    let mut ranking: Vec<usize> = (0..model::SKILL_COUNT).collect();
    ranking.sort_by(|&a, &b| focus_sum[b].partial_cmp(&focus_sum[a]).unwrap());

    let mut blocks_json = Vec::new();
    for &(skill, start, count) in &blocks {
        blocks_json.push(format!(
            r#"{{"skill":{},"name":"{}","weekStart":{},"weeks":{}}}"#,
            skill,
            json_esc(model::SKILL_NAMES[skill]),
            start,
            count
        ));
    }
    let mut rank_json = Vec::new();
    for (i, &s) in ranking.iter().enumerate().take(model::SKILL_COUNT) {
        rank_json.push(format!(
            r#"{{"rank":{},"skill":{},"name":"{}","focus":{:.3}}}"#,
            i,
            s,
            json_esc(model::SKILL_NAMES[s]),
            focus_sum[s]
        ));
    }

    let weeks_to_press = result
        .trace
        .iter()
        .position(|&t| t >= evo::PRESS_THRESHOLD)
        .map(|p| p + 1)
        .unwrap_or(0); // 0 = not reached within horizon

    let start_caps = caps;
    let start_at = model::attainment(&start_caps);

    format!(
        r#"{{"caps":{},"startAttainment":{:.3},"finalAttainment":{:.3},"weeksToPress":{},"injuries":{},"source":"{}","evoAttainment":{:.3},"baseAttainment":{:.3},"gen":{},"trace":{},"blocks":[{}],"ranking":[{}]}}"#,
        floats_json(&start_caps),
        start_at,
        result.final_score,
        weeks_to_press,
        result.injuries,
        source,
        evo_score,
        base_score,
        ev.gen,
        floats_json(&result.trace),
        blocks_json.join(","),
        rank_json.join(","),
    )
}

fn handle(
    mut stream: TcpStream,
    evolution: Arc<Mutex<evo::Evolution>>,
    running: Arc<AtomicBool>,
) {
    let req = match read_request(&mut stream) {
        Some(r) => r,
        None => return,
    };

    let path = req.path.split('?').next().unwrap_or("/").to_string();

    if req.method == "OPTIONS" {
        write_response(&mut stream, "204 No Content", "text/plain", "");
        return;
    }

    match (req.method.as_str(), path.as_str()) {
        ("GET", "/api/evolution") => respond_json(&mut stream, &api_evolution(&evolution, &running)),
        ("GET", "/api/meta") => respond_json(&mut stream, &api_meta()),
        ("POST", "/api/recommend") => {
            respond_json(&mut stream, &api_recommend(&evolution, &req.body))
        }
        ("POST", "/api/control") => {
            let run = req.body.contains("true");
            running.store(run, Ordering::Relaxed);
            respond_json(&mut stream, &format!(r#"{{"running":{run}}}"#))
        }
        ("GET", "/") => serve_static(&mut stream, "/index.html"),
        _ => serve_static(&mut stream, &path),
    };
}
