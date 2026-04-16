const CORRECT_PASSWORD = 'boom261#';

let matchCount = 0;

function getReliability(coteFav) {
  if (coteFav < 1.5) return { stars: '★★★★★', label: 'Très fort' };
  if (coteFav < 1.9) return { stars: '★★★★☆', label: 'Fort' };
  if (coteFav < 2.3) return { stars: '★★★☆☆', label: 'Moyen' };
  return { stars: '★★☆☆☆', label: 'Risqué' };
}

function analyzeMatch(card) {
  const coteA = parseFloat(card.querySelector('.coteA').value);
  const coteB = parseFloat(card.querySelector('.coteB').value);

  const teamA = card.querySelector('.teamA').value;
  const teamB = card.querySelector('.teamB').value;
  const time = card.querySelector('.match-time').value;

  if (isNaN(coteA) || isNaN(coteB)) return null;

  const coteFav = Math.min(coteA, coteB);

  let outcome, score, total;

  if (coteFav < 1.50) {
    outcome = coteA < coteB ? '1' : '2';
    score = '2-0';
    total = 2;
  } 
  else if (coteFav <= 2.20) {
    outcome = coteA < coteB ? '1' : '2';
    score = '2-1';
    total = 3;
  } 
  else {
    outcome = 'X';
    score = '2-2';
    total = 4;
  }

  return {
    teamA,
    teamB,
    time,
    outcome,
    score,
    total,
    rel: getReliability(coteFav)
  };
}

/* ─── ANALYSE BUTTON ─── */
document.getElementById('analyzeBtn').addEventListener('click', () => {

  const cards = document.querySelectorAll('.match-card');
  const errorEl = document.getElementById('errorMsg');
  const resultsSection = document.getElementById('resultsSection');
  const resultsCards = document.getElementById('resultsCards');

  if (cards.length === 0) {
    errorEl.textContent = '⚠️ Ajoutez au moins un match.';
    errorEl.classList.add('show');
    return;
  }

  let html = '';
  let hasError = false;

  cards.forEach(card => {
    const res = analyzeMatch(card);
    if (!res) { hasError = true; return; }

    html += `
      <div class="result-card">
        <div class="rc-header">
          <div>
            <span class="boost-badge">⭐ BOOST MODE</span>
            <div class="rc-matchup">${res.teamA} vs ${res.teamB}</div>
          </div>
          <div class="rc-time">🕐 ${res.time}</div>
        </div>

        <div class="stats-row">
          <div class="stat-box v1x2">
            <div class="stat-label">1X2</div>
            <div class="stat-value">${res.outcome}</div>
          </div>

          <div class="stat-box vscore">
            <div class="stat-label">Score</div>
            <div class="stat-value">${res.score}</div>
          </div>

          <div class="stat-box vtotal">
            <div class="stat-label">Total</div>
            <div class="stat-value">${res.total}</div>
          </div>
        </div>

        <div class="reliability">
          <span class="stars">${res.rel.stars}</span>
          <span class="label">${res.rel.label}</span>
        </div>
      </div>
    `;
  });

  if (hasError) {
    errorEl.textContent = '⚠️ Remplis toutes les cotes correctement.';
    errorEl.classList.add('show');
    return;
  }

  errorEl.classList.remove('show');
  resultsCards.innerHTML = html;
  resultsSection.classList.add('show');
  resultsSection.scrollIntoView({ behavior: 'smooth' });
});

/* ─── LOGIN ─── */
document.getElementById('loginBtn').addEventListener('click', () => {
  const pwd = document.getElementById('loginPassword').value;

  if (pwd === CORRECT_PASSWORD) {
    document.getElementById('loginScreen').style.display = 'none';
    document.getElementById('mainApp').classList.add('show');
    addMatchCard('Manchester City', 'Sunderland', '07:00');
  } else {
    document.getElementById('loginError').classList.add('show');
  }
});

/* ─── ADD MATCH ─── */
document.getElementById('addMatchBtn').addEventListener('click', () => {
  addMatchCard();
});

/* ─── ADD MATCH CARD ─── */
function addMatchCard(teamA='', teamB='', time='07:00') {
  matchCount++;

  const card = document.createElement('div');
  card.className = 'match-card';
  card.dataset.id = matchCount;

  card.innerHTML = `
    <div class="card-header">
      <span class="match-label">Match ${matchCount}</span>
    </div>

    <div class="time-row">
      <label>Heure</label>
      <input type="time" class="match-time" value="${time}">
    </div>

    <div class="teams-row">
      <select class="teamA">
        <option>${teamA || "Team A"}</option>
      </select>

      <select class="teamB">
        <option>${teamB || "Team B"}</option>
      </select>
    </div>

    <div class="odds-row">
      <input class="coteA" type="number" placeholder="Cote 1">
      <input class="coteB" type="number" placeholder="Cote 2">
    </div>
  `;

  document.getElementById('matchesWrapper').appendChild(card);
}