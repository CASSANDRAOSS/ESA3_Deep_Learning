let vocab = [];
let word2idx = {};
let idx2word = {};
let sequences = [];
let nextWords = [];
const sequenceLength = 5;
let model;
let modelReady = false;

// UI Elemente
const predictBtn = document.getElementById("predictBtn");
const nextBtn = document.getElementById("nextBtn");
const autoBtn = document.getElementById("autoBtn");
const stopBtn = document.getElementById("stopBtn");
const resetBtn = document.getElementById("resetBtn");
const statusDiv = document.getElementById("status");
const resultsDiv = document.getElementById("results");

function setButtonsEnabled(enabled) {
    [predictBtn, nextBtn, autoBtn, stopBtn, resetBtn].forEach(btn => btn.disabled = !enabled);
}

// 1. DATEN LADEN
async function loadData() {
    const response = await fetch('data/text_corpus.txt');
    const text = await response.text();
    const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 0);

    vocab = Array.from(new Set(words));
    vocab.forEach((word, idx) => {
        word2idx[word] = idx;
        idx2word[idx] = word;
    });

    for (let i = 0; i <= words.length - sequenceLength - 1; i++) {
        const seq = words.slice(i, i + sequenceLength);
        sequences.push(seq.map(w => word2idx[w]));
        nextWords.push(word2idx[words[i + sequenceLength]]);
    }
}

// 2. TENSOREN ERSTELLEN
function prepareTrainingData() {
    const X = sequences.map(seq => 
        seq.map(idx => {
            const oneHot = new Array(vocab.length).fill(0);
            oneHot[idx] = 1;
            return oneHot;
        })
    );
    const y = nextWords.map(idx => {
        const vec = new Array(vocab.length).fill(0);
        vec[idx] = 1;
        return vec;
    });

    return { X_tensor: tf.tensor3d(X), y_tensor: tf.tensor2d(y) };
}

// 3. MODELL (Stacked LSTM)
function createModel() {
    model = tf.sequential();
    model.add(tf.layers.lstm({
        units: 100,
        returnSequences: true,
        inputShape: [sequenceLength, vocab.length]
    }));
    model.add(tf.layers.lstm({ units: 100 }));
    model.add(tf.layers.dense({ units: vocab.length, activation: 'softmax' }));

    model.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
}

async function trainModel(X, y) {
    const callbacks = tfvis.show.fitCallbacks({ name: 'Training', tab: 'Status' }, ['loss', 'acc']);
    await model.fit(X, y, { epochs: 50, batchSize: 32, shuffle: true, callbacks });
}

// 4. VORHERSAGE
function predictNextWord(inputText, topK = 5) {
    if (!modelReady) return [];
    const words = inputText.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/).filter(w => w.length > 0);
    if (words.length < sequenceLength) return [];

    const lastWords = words.slice(-sequenceLength);
    const inputSeq = lastWords.map(w => word2idx[w] ?? Math.floor(Math.random() * vocab.length));

    return tf.tidy(() => {
        const inputIdx = inputSeq.map(idx => {
            const oh = new Array(vocab.length).fill(0);
            oh[idx] = 1;
            return oh;
        });
        const inputTensor = tf.tensor3d([inputIdx]);
        const probs = model.predict(inputTensor).dataSync();
        
        return Array.from(probs.keys())
            .sort((a, b) => probs[b] - probs[a])
            .slice(0, topK)
            .map(idx => ({ word: idx2word[idx], probability: probs[idx] }));
    });
}

// 5. EVALUATION (Async zur Performance)
async function evaluateModel(X, y) {
    const total = X.shape[0];
    let top1 = 0, top5 = 0;
    let lossSum = 0;

    for (let i = 0; i < total; i++) {
        if (i % 20 === 0) await tf.nextFrame(); // Browser-Pause

        const input = X.slice([i, 0, 0], [1, sequenceLength, vocab.length]);
        const trueIdx = y.slice([i, 0], [1, vocab.length]).argMax(-1).dataSync()[0];
        
        const probs = model.predict(input).dataSync();
        const sorted = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);

        if (sorted[0] === trueIdx) top1++;
        if (sorted.slice(0, 5).includes(trueIdx)) top5++;
        lossSum += -Math.log(probs[trueIdx] + 1e-7);
    }

    resultsDiv.innerHTML = `
        <p>Top-1 Accuracy: ${(top1/total*100).toFixed(2)}%</p>
        <p>Top-5 Accuracy: ${(top5/total*100).toFixed(2)}%</p>
        <p>Perplexity: ${Math.exp(lossSum/total).toFixed(2)}</p>
    `;
}

// PIPELINE START
async function runTraining() {
    setButtonsEnabled(false);
    statusDiv.textContent = "Lade Korpus...";
    await loadData();
    
    statusDiv.textContent = "Bereite Tensoren vor...";
    const { X_tensor, y_tensor } = prepareTrainingData();
    
    createModel();
    statusDiv.textContent = "Training läuft (siehe Visor)...";
    await trainModel(X_tensor, y_tensor);
    
    if (window.innerWidth < 600) tfvis.visor().close();
    
    statusDiv.textContent = "Auswertung...";
    await evaluateModel(X_tensor, y_tensor);
    
    statusDiv.textContent = "Bereit!";
    modelReady = true;
    setButtonsEnabled(true);
}

// UI LOGIK
predictBtn.onclick = () => displayPredictions(predictNextWord(document.getElementById("inputText").value));

function displayPredictions(preds) {
    const div = document.getElementById("predictions");
    div.innerHTML = "";
    preds.forEach(p => {
        const b = document.createElement("button");
        b.textContent = `${p.word} (${(p.probability*100).toFixed(1)}%)`;
        b.onclick = () => {
            document.getElementById("inputText").value += " " + p.word;
            displayPredictions(predictNextWord(document.getElementById("inputText").value));
        };
        div.appendChild(b);
    });
}

nextBtn.onclick = () => {
    const area = document.getElementById("inputText");
    const p = predictNextWord(area.value, 1);
    if (p.length > 0) {
        area.value += " " + p[0].word;
        displayPredictions(predictNextWord(area.value));
    }
};

resetBtn.onclick = () => {
    document.getElementById("inputText").value = "";
    document.getElementById("predictions").innerHTML = "";
};

document.querySelectorAll(".dropdown-toggle").forEach(b => {
    b.onclick = () => b.parentElement.classList.toggle("open");
});

runTraining();
