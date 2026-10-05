const { porterStem } = require("./porter_stemmer_node_version.js");

// ---- Stopwords (same set the Python side uses, minus "not"/"no") ----
const STOPWORDS = new Set([
  "the","a","an","is","are","was","were","be","been","being","am","i","you",
  "he","she","it","we","they","them","him","her","us","my","your","his",
  "its","our","their","of","to","in","on","at","for","with","and","or",
  "but","if","so","this","that","these","those","do","does","did","have",
  "has","had","having","will","would","can","could","should","shall","may",
  "might","must","about","above","after","again","against","all","also",
  "any","because","before","below","between","both","by","down","during",
  "each","few","from","further","here","how","into","more","most","now",
  "once","only","other","out","over","own","same","some","such","than",
  "then","there","through","too","under","until","up","very","what","when",
  "where","which","while","who","whom","why","yourself","yourselves",
  "myself","himself","herself","itself","ourselves","themselves","as",
  "be","being","been","doing","don","should've","ll","re","ve","y","ain",
  "aren","couldn","didn","doesn","hadn","hasn","haven","isn","ma","mightn",
  "mustn","needn","shan","shouldn","wasn","weren","won","wouldn",
]);
// "not" and "no" are deliberately kept (matches AgroVox's _KEEP_WORDS)

function isTamil(text) {
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (cp >= 0x0b80 && cp <= 0x0bff) return true;
  }
  return false;
}

function detectLanguage(text) {
  if (!text) return "en";
  let tamil = 0, letters = 0;
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (cp >= 0x0b80 && cp <= 0x0bff) tamil++;
    if (/\p{L}/u.test(ch)) letters++;
  }
  if (letters === 0) return "en";
  return tamil / letters > 0.3 ? "ta" : "en";
}

function cleanText(text) {
  return text
    .replace(/https?:\/\/\S+|www\.\S+/gi, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

// Lemmatization-free approximation: AgroVox's real pipeline lemmatizes then
// stems; a browser demo skips WordNet lemmatization (no offline corpus in
// JS) and relies on stemming alone to collapse inflectional variants. This
// is documented as a deliberate Demo Mode simplification.
function stemmedFeatureTokens(text) {
  const cleaned = cleanText(text);
  const rawTokens = cleaned.match(/\b[\w]{2,}\b/g) || [];
  const kept = rawTokens.filter(t => !STOPWORDS.has(t));
  return kept.map(porterStem);
}

function buildNgrams(tokens, ngramRange) {
  const [lo, hi] = ngramRange;
  const grams = [];
  for (let n = lo; n <= hi; n++) {
    for (let i = 0; i + n <= tokens.length; i++) {
      grams.push(tokens.slice(i, i + n).join(" "));
    }
  }
  return grams;
}

// Vectorize into a sparse {index: tfidfValue} map, replicating sklearn's
// TfidfVectorizer(sublinear_tf=True, norm='l2').
function vectorizeTfidf(tokens, vocab, idf, ngramRange) {
  const grams = buildNgrams(tokens, ngramRange);
  const counts = {};
  for (const g of grams) {
    if (Object.prototype.hasOwnProperty.call(vocab, g)) {
      counts[g] = (counts[g] || 0) + 1;
    }
  }
  const raw = {}; // index -> value (pre-normalization)
  for (const [term, count] of Object.entries(counts)) {
    const idx = vocab[term];
    const tf = 1 + Math.log(count); // sublinear_tf
    raw[idx] = tf * idf[idx];
  }
  let normSq = 0;
  for (const v of Object.values(raw)) normSq += v * v;
  const norm = Math.sqrt(normSq) || 1;
  const vec = {};
  for (const [idx, v] of Object.entries(raw)) vec[idx] = v / norm;
  return vec;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

function classifyIntent(text, language, classifierData) {
  const { vocabulary, idf, ngram_range, classes, model_type,
          class_log_prior, feature_log_prob, coef, intercept } = classifierData;
  const tokens = stemmedFeatureTokens(text);
  const vec = vectorizeTfidf(tokens, vocabulary, idf, ngram_range);

  let scores;
  if (model_type === "naive_bayes") {
    scores = classes.map((_, c) => {
      let s = class_log_prior[c];
      for (const [idx, val] of Object.entries(vec)) {
        s += val * feature_log_prob[c][idx];
      }
      return s;
    });
  } else {
    scores = classes.map((_, c) => {
      let s = intercept[c];
      for (const [idx, val] of Object.entries(vec)) {
        s += val * coef[c][idx];
      }
      return s;
    });
  }
  const probs = softmax(scores);
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;
  return { intent: classes[bestIdx], confidence: probs[bestIdx], tokens };
}

function cosineTopMatch(text, qaData) {
  const tokens = stemmedFeatureTokens(text);
  const vec = vectorizeTfidf(tokens, qaData.vocabulary, qaData.idf, [1, 1]);
  let bestScore = -1, bestIdx = -1;
  qaData.doc_vectors.forEach((docVec, i) => {
    let score = 0;
    for (const [idx, val] of Object.entries(vec)) {
      score += val * (docVec[idx] || 0);
    }
    if (score > bestScore) { bestScore = score; bestIdx = i; }
  });
  if (bestIdx === -1) return null;
  return {
    question: qaData.questions[bestIdx],
    answer: qaData.answers[bestIdx],
    category: qaData.categories[bestIdx],
    confidence: bestScore,
  };
}

// ---- Machine Translation: greedy longest-phrase-first dictionary lookup,
// mirroring AgroVox's DictionaryTranslator._phrase_translate exactly
// (whitespace-tokenized, never \w+/\b on the Tamil side). ----
function phraseTranslate(text, vocab, lowercaseKeys, maxPhraseLen) {
  const tokens = text.split(/\s+/).filter(Boolean);
  const n = tokens.length;
  const output = [];
  let i = 0;
  const PUNCT = /[!"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]/g;
  while (i < n) {
    let matched = false;
    const maxLen = Math.min(maxPhraseLen, n - i);
    for (let length = maxLen; length >= 1; length--) {
      const window = tokens.slice(i, i + length);
      const cleanWindow = window.map(w => w.replace(PUNCT, ""));
      if (cleanWindow.some(w => w === "")) continue;
      const key = lowercaseKeys
        ? cleanWindow.map(w => w.toLowerCase()).join(" ")
        : cleanWindow.join(" ");
      if (Object.prototype.hasOwnProperty.call(vocab, key)) {
        const trailing = (window[window.length - 1].match(PUNCT) || []).join("");
        output.push(vocab[key] + trailing);
        i += length;
        matched = true;
        break;
      }
    }
    if (!matched) { output.push(tokens[i]); i += 1; }
  }
  return output.join(" ");
}

function translateTaToEn(text, translationDict) {
  const vocab = translationDict.ta_to_en;
  const maxLen = Math.max(1, ...Object.keys(vocab).map(k => k.split(" ").length));
  return phraseTranslate(text, vocab, false, maxLen);
}

function translateEnToTa(text, translationDict) {
  const vocab = translationDict.en_to_ta;
  const maxLen = Math.max(1, ...Object.keys(vocab).map(k => k.split(" ").length));
  return phraseTranslate(text, vocab, true, maxLen);
}

// ---- Named Entity Recognition: gazetteer longest-match, case-insensitive
// for Latin script, exact for Tamil script. ----
function extractEntities(text, gazetteer) {
  const entries = [];
  for (const [type, terms] of Object.entries(gazetteer)) {
    for (const term of terms) {
      if (!term) continue;
      entries.push({ term, type, len: term.length });
    }
  }
  entries.sort((a, b) => b.len - a.len); // longest terms matched first

  const lowerText = text.toLowerCase();
  const found = [];
  const claimed = new Array(text.length).fill(false);

  for (const { term, type } of entries) {
    const isAscii = /^[\x00-\x7f\s]+$/.test(term);
    const hay = isAscii ? lowerText : text;
    const needle = isAscii ? term.toLowerCase() : term;
    let searchFrom = 0;
    while (true) {
      const idx = hay.indexOf(needle, searchFrom);
      if (idx === -1) break;
      searchFrom = idx + needle.length;
      const overlaps = claimed.slice(idx, idx + needle.length).some(Boolean);
      if (overlaps) continue;
      // word-boundary check for ASCII terms (avoid matching inside a longer word)
      if (isAscii) {
        const before = idx > 0 ? hay[idx - 1] : " ";
        const after = idx + needle.length < hay.length ? hay[idx + needle.length] : " ";
        if (/[a-z0-9]/.test(before) || /[a-z0-9]/.test(after)) continue;
      }
      for (let k = idx; k < idx + needle.length; k++) claimed[k] = true;
      const wordCount = term.trim().split(/\s+/).length;
      const confidence = wordCount >= 2 ? 0.97 : 0.91; // rule-based match, not a model probability
      found.push({ text: text.slice(idx, idx + needle.length), type, confidence, start: idx });
    }
  }
  return found.sort((a, b) => a.start - b.start);
}

module.exports = {
  detectLanguage, isTamil, stemmedFeatureTokens, classifyIntent, cosineTopMatch,
  translateTaToEn, translateEnToTa, extractEntities,
};
