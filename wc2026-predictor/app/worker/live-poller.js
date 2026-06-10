/**
 * Cloudflare Worker: central live-score poller (framework v3 §C.1).
 *
 * Cron trigger (1-2 min during match windows) fetches the scoreboard ONCE,
 * normalizes it, and stores data/live.json in KV. A million visitors then
 * poll OUR cached JSON; upstream sees ~1 req/min total.
 *
 * Primary source: ESPN unofficial scoreboard JSON (keyless; undocumented —
 * polled centrally only, never from browsers). Licensed fallback:
 * football-data.org free tier (set FOOTBALL_DATA_TOKEN secret + USE_FALLBACK=1).
 *
 * Deploy:  cd app/worker && npx wrangler deploy
 * KV:      npx wrangler kv namespace create LIVE   (bind as LIVE in wrangler.toml)
 */

const ESPN_URL =
  "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard";
const FD_URL = "https://api.football-data.org/v4/competitions/WC/matches?status=LIVE,IN_PLAY,PAUSED";

async function fetchEspn() {
  const r = await fetch(ESPN_URL, { headers: { "User-Agent": "wc26-predictor-poller" } });
  if (!r.ok) throw new Error(`espn ${r.status}`);
  const j = await r.json();
  return (j.events || []).map((e) => {
    const c = e.competitions?.[0];
    const [home, away] = (c?.competitors || []).sort((x) => (x.homeAway === "home" ? -1 : 1));
    return {
      home: home?.team?.displayName, away: away?.team?.displayName,
      home_score: home?.score, away_score: away?.score,
      status: e.status?.type?.shortDetail || "",
    };
  });
}

async function fetchFallback(env) {
  const r = await fetch(FD_URL, { headers: { "X-Auth-Token": env.FOOTBALL_DATA_TOKEN } });
  if (!r.ok) throw new Error(`football-data ${r.status}`);
  const j = await r.json();
  return (j.matches || []).map((m) => ({
    home: m.homeTeam?.name, away: m.awayTeam?.name,
    home_score: m.score?.fullTime?.home ?? m.score?.halfTime?.home ?? 0,
    away_score: m.score?.fullTime?.away ?? m.score?.halfTime?.away ?? 0,
    status: m.status,
  }));
}

export default {
  // cron: poll + store
  async scheduled(_event, env, _ctx) {
    let matches = [];
    try {
      matches = env.USE_FALLBACK === "1" ? await fetchFallback(env) : await fetchEspn();
    } catch (e) {
      try { matches = await fetchFallback(env); } catch { /* keep last good */ }
    }
    if (matches.length || !(await env.LIVE.get("live"))) {
      await env.LIVE.put("live", JSON.stringify({ updated: new Date().toISOString(), matches }));
    }
  },

  // serve /data/live.json from KV (also CORS-open for embeds)
  async fetch(req, env) {
    const url = new URL(req.url);
    if (url.pathname.endsWith("/live.json")) {
      const body = (await env.LIVE.get("live")) || JSON.stringify({ updated: null, matches: [] });
      return new Response(body, {
        headers: {
          "content-type": "application/json",
          "cache-control": "public, max-age=30, stale-while-revalidate=120",
          "access-control-allow-origin": "*",
        },
      });
    }
    return new Response("wc26 live poller", { status: 200 });
  },
};
