# Tunewise — RL-Powered Music Playlist Curator

A contextual bandit music recommendation system that learns individual user preferences in real time using **LinUCB**, a reinforcement learning algorithm. The agent improves with every skip, completion, and favourite — no collaborative filtering, no static popularity rankings.

---

## The Problem

Mainstream streaming algorithms create filter bubbles and cannot adapt to real-time mood or context. Skip signals are rich feedback that most systems discard.

## The Solution

A per-user **LinUCB contextual bandit** agent learns audio feature preferences (energy, valence, danceability, tempo, etc.) from skip/completion signals. Each user gets their own agent stored in the database — their recommendations improve independently and never mix with other users.

---

## Repository Structure

```
music-rl-curator/
├── backend/          (main branch)  FastAPI backend + agent logic
├── frontend/         (frontend branch)  React/Node version of the UI
├── frontend_html/    (frontend_html branch)  Standalone HTML version (no build needed)
└── linucb/           (linucb branch)  Jupyter notebooks — data cleaning, simulation, training
```

---

## Branches

### `main` — Backend

FastAPI REST API with per-user LinUCB agents stored in PostgreSQL (Neon).

```
backend/
├── main.py          # All API endpoints
├── agent.py         # LinUCBAgent class — select(), update(), load/save to DB
├── context.py       # Builds 12-dim feature vectors (8 audio + 4 context)
├── models.py        # SQLAlchemy models: User, AgentState, ListenHistory
├── database.py      # Neon PostgreSQL connection with pool_pre_ping
├── clean.csv        # Cleaned Spotify dataset (~89k tracks)
├── scaler.pkl       # Fitted MinMaxScaler for 8 audio features
└── requirements.txt
```

**Key endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/register` | Create account |
| POST | `/login` | Sign in |
| GET | `/cold-start` | 20 diverse songs for new users |
| GET | `/recommend-many/{user_id}` | 15 RL-ranked recommendations |
| GET | `/search/{user_id}` | Search by song/artist |
| POST | `/feedback` | Send reward signal — agent learns here |
| GET | `/history/{user_id}` | User listen history |
| GET | `/agent-stats/{user_id}` | Learned θ weights per feature |
| GET | `/admin/stats` | All users — rewards, preferences, breakdowns |

**Reward signals:**

| Action | Reward |
|--------|--------|
| Completed | +2.0 |
| Favourited | +3.0 |
| Skipped after 50% | +0.5 |
| Skipped early | −1.0 |
| Removed from queue | −2.0 |

---

### `linucb` — Notebooks

Data cleaning and RL training simulation. Run in Google Colab.

**Notebook 1 — Data Cleaning**
- Loads raw [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) (~114k tracks)
- Drops duplicates, nulls, zero-tempo and zero-popularity tracks
- Fits `MinMaxScaler` on 8 audio features
- Outputs: `spotify_clean.csv`, `feature_matrix.npy`, `scaler.pkl`

**Notebook 2 — LinUCB Training & Evaluation**
- Implements `LinUCBAgent` (12-dim: 8 audio + 4 context features)
- `SimulatedUser` with noise and 5 reward types
- 2000 episode training loop
- Baseline comparison: Random vs Popularity vs LinUCB
- Metrics: diversity, serendipity, skip analytics
- Outputs: `linucb_agent.pkl`, reward logs

**Results:**

| Method | Avg Reward |
|--------|-----------|
| Random shuffle | 1.242 |
| Popularity-based | 1.569 |
| **LinUCB RL** | **1.835** |

LinUCB outperforms popularity-based recommendations by **+16.9%** and random by **+47.7%**.

---

### `frontend_html` — Standalone HTML UI

Single-file Tunewise UI. No build step, no npm — open directly in browser.

```
tunewise.html    # Complete app in one file
```

**Features:**
- Login / Register
- Cold start: 20 diverse songs on first visit (picks across 10 genres)
- Now-playing bar with 5 feedback action buttons
- Spotify-style recommendation list that refreshes after each feedback
- Search bar with live results
- Listen history page
- Admin dashboard (login: `admin` / `admin123`):
  - Reward-over-interactions line chart per user
  - Action breakdown donut chart
  - Avg reward bar chart per user
  - Per-user agent detail panel with 8 feature preference cards

---

### `frontend` — React/Node UI

React version of the same UI. Requires Node.js.

```bash
cd frontend
npm install
npm run dev
```

---

## State & Feature Space

The agent operates on a **12-dimensional feature vector** per song:

| Index | Feature | Source |
|-------|---------|--------|
| 0 | Danceability | Spotify audio feature (scaled) |
| 1 | Energy | Spotify audio feature (scaled) |
| 2 | Valence (mood) | Spotify audio feature (scaled) |
| 3 | Tempo (BPM) | Spotify audio feature (scaled) |
| 4 | Loudness | Spotify audio feature (scaled) |
| 5 | Acousticness | Spotify audio feature (scaled) |
| 6 | Instrumentalness | Spotify audio feature (scaled) |
| 7 | Speechiness | Spotify audio feature (scaled) |
| 8 | Time of day | Runtime: `hour / 24.0` |
| 9 | Activity | Runtime: 0.0=relax / 0.5=work / 1.0=exercise |
| 10 | Session skip rate | Runtime: from DB history |
| 11 | Recency penalty | Runtime: played in last 20? |

Genre is deliberately **excluded** from the feature vector — the 8 audio features already encode genre implicitly (high tempo + energy ≈ EDM, high acousticness + low energy ≈ folk). Genre is only used for cold-start diversity.

---

## Algorithm — LinUCB Contextual Bandit

```
At each step:
  1. Sample 200 candidate songs randomly
  2. Build 12-dim context vectors for each
  3. Score each: score = θᵀx + α√(xᵀA⁻¹x)
                         ↑ exploit   ↑ explore
  4. Split into: 50% exploit (top predicted reward)
                 30% explore (highest uncertainty)
                 20% random buffer
  5. Return top 15 after Deezer preview fetch
  6. On user feedback: A += xxᵀ, b += r·x
```

`A` (12×12) and `b` (12,) are stored as JSON in the `agent_states` table — one row per user.

---

## Running Locally

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
DATABASE_URL=postgresql://user:password@your-neon-host/neondb
```

```bash
uvicorn main:app --reload
# API at http://127.0.0.1:8000
# Docs at http://127.0.0.1:8000/docs
```

### Frontend (HTML version)

Just open `tunewise.html` in Chrome. Make sure the backend is running at `http://127.0.0.1:8000`.

### Frontend (React version)

```bash
cd frontend
npm install
npm run dev
```

---

## Database Schema (Neon PostgreSQL)

```sql
users (id TEXT PK, username TEXT UNIQUE, password_hash TEXT, created_at TIMESTAMP)

agent_states (user_id TEXT PK, A_matrix TEXT, b_vector TEXT, updated_at TIMESTAMP)
-- A_matrix: JSON string of 12×12 numpy array
-- b_vector: JSON string of 12-dim numpy array

listen_history (id SERIAL PK, user_id TEXT, track_id TEXT,
                track_name TEXT, reward FLOAT, played_at TIMESTAMP)
```

---

## Dataset

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — 114k tracks across 114 genres.

After cleaning: ~89,740 tracks (removed duplicates, nulls, zero-tempo, zero-popularity).

Audio preview URLs served via [Deezer API](https://developers.deezer.com/api) (free, no auth required). Results are cached in-memory to avoid redundant API calls.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| RL Algorithm | LinUCB Contextual Bandit (NumPy) |
| Backend | FastAPI + SQLAlchemy |
| Database | PostgreSQL via Neon (serverless) |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| Frontend (alt) | React + Vite |
| Audio previews | Deezer API |
| Dataset | Kaggle Spotify Tracks |
| Notebooks | Google Colab |

---

## Requirements

```
fastapi
uvicorn
sqlalchemy
psycopg2-binary
passlib[bcrypt]
python-dotenv
pandas
numpy
scikit-learn
requests
pydantic
```
