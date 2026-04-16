document.addEventListener('DOMContentLoaded', () => {
    
    // Éléments du Bankroll
    const bankrollInp = document.getElementById('bankroll');
    const safeBetEl = document.getElementById('safeBet');
    const maxBetEl = document.getElementById('maxBet');

    // Éléments de l'Analyse
    const coteAInp = document.getElementById('coteA');
    const coteBInp = document.getElementById('coteB');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const errorMsg = document.getElementById('errorMsg');

    // Éléments des Résultats
    const resultsSection = document.getElementById('resultsSection');
    const resFavori = document.getElementById('resFavori');
    const resPari = document.getElementById('resPari');
    const resScore = document.getElementById('resScore');
    const cardSecurity = document.getElementById('cardSecurity');

    // Formateur de devise
    const formatter = new Intl.NumberFormat('fr-FR');

    // 1. Logique Bankroll
    function updateBankroll() {
        const capital = parseFloat(bankrollInp.value);
        if(isNaN(capital) || capital <= 0) {
            safeBetEl.textContent = "0";
            maxBetEl.textContent = "0";
            return;
        }

        const safeBet = capital * 0.05; // 5%
        const maxBet = capital * 0.10;  // 10%

        safeBetEl.textContent = formatter.format(safeBet);
        maxBetEl.textContent = formatter.format(maxBet);
    }

    bankrollInp.addEventListener('input', updateBankroll);

    // 2. Algorithme d'analyse de match
    function analyzeMatch() {
        const coteA = parseFloat(coteAInp.value);
        const coteB = parseFloat(coteBInp.value);

        if(isNaN(coteA) || isNaN(coteB) || coteA <= 1 || coteB <= 1) {
            errorMsg.classList.remove('hidden');
            resultsSection.classList.add('hidden');
            return;
        }

        errorMsg.classList.add('hidden');

        // Etape 1: Déterminer le favori
        const teamASelect = document.getElementById('teamA');
        const teamBSelect = document.getElementById('teamB');
        let nomEquipeA = (teamASelect && teamASelect.value !== "") ? teamASelect.value : "Équipe A";
        let nomEquipeB = (teamBSelect && teamBSelect.value !== "") ? teamBSelect.value : "Équipe B";

        let favoriString = "";
        let coteFavori = 0;

        if (coteA < coteB) {
            favoriString = nomEquipeA;
            coteFavori = coteA;
        } else if (coteB < coteA) {
            favoriString = nomEquipeB;
            coteFavori = coteB;
        } else {
            favoriString = "Aucun (Égalité parfaite)";
            coteFavori = coteA;
        }

        // Etape 2: Calculer la différence
        const diff = Math.abs(coteA - coteB);

        // Etape 3: Prédiction
        let scoreProbable = "";
        let pariConseille = "";

        if (coteFavori < 1.50) {
            scoreProbable = "2-0 ou 3-0";
            pariConseille = "Over 1.5";
        } else if (coteFavori >= 1.50 && coteFavori <= 2.20) {
            scoreProbable = "2-1 ou 1-0";
            pariConseille = "Over 2.5";
        } else {
            // > 2.20
            scoreProbable = "2-2 ou 3-1";
            pariConseille = "Over 3.5";
        }

        // MAJ DOM
        resFavori.textContent = favoriString + " (" + coteFavori + ")";
        resPari.textContent = pariConseille;
        resScore.textContent = scoreProbable;

        // Etape 4: Sécurité
        if (diff < 0.30) {
            cardSecurity.classList.remove('hidden');
        } else {
            cardSecurity.classList.add('hidden');
        }

        // Afficher les résultats avec petit effet
        resultsSection.classList.remove('hidden');
        resultsSection.style.opacity = 0;
        
        setTimeout(() => {
            resultsSection.style.transition = "opacity 0.5s ease-in-out";
            resultsSection.style.opacity = 1;
        }, 50);

        // Scroll doux vers les résultats
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    analyzeBtn.addEventListener('click', analyzeMatch);
    
    // Optionnel : Lancer au bouton "Entrée"
    [coteAInp, coteBInp].forEach(input => {
        input.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                analyzeMatch();
            }
        });
    });

    // Optionnel : Fausse action pour l'upload d'image
    const screenshotBtn = document.getElementById('screenshotBtn');
    screenshotBtn.addEventListener('change', function(e) {
        if(e.target.files.length > 0) {
            alert("Fonctionnalité d'extraction de données par image en cours de développement. Le fichier "+e.target.files[0].name+" a été reçu.");
        }
    });

});
