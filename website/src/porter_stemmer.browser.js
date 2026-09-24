// Classic Porter Stemmer (1980), plain JS, no dependencies.
function porterStem(w) {
  w = w.toLowerCase();
  if (w.length < 3) return w;

  const isVowel = (ch) => "aeiou".includes(ch);
  const isConsonant = (word, i) => {
    const ch = word[i];
    if (isVowel(ch)) return false;
    if (ch === "y") {
      return i === 0 ? true : !isConsonant(word, i - 1);
    }
    return true;
  };

  const measure = (stem) => {
    let n = 0, i = 0;
    const len = stem.length;
    while (i < len) {
      while (i < len && isConsonant(stem, i)) i++;
      if (i >= len) break;
      while (i < len && !isConsonant(stem, i)) i++;
      n++;
      while (i < len && isConsonant(stem, i)) i++;
    }
    return n;
  };
  const m = (stem) => {
    // measure = number of VC sequences
    let count = 0, i = 0;
    const len = stem.length;
    // skip leading consonants
    while (i < len && isConsonant(stem, i)) i++;
    while (i < len) {
      // skip vowels
      while (i < len && !isConsonant(stem, i)) i++;
      if (i >= len) break;
      // skip consonants
      while (i < len && isConsonant(stem, i)) i++;
      count++;
    }
    return count;
  };
  const containsVowel = (stem) => {
    for (let i = 0; i < stem.length; i++) if (!isConsonant(stem, i)) return true;
    return false;
  };
  const endsWithDoubleCons = (stem) => {
    const len = stem.length;
    if (len < 2) return false;
    return stem[len - 1] === stem[len - 2] && isConsonant(stem, len - 1);
  };
  const endsWithCVC = (stem) => {
    const len = stem.length;
    if (len < 3) return false;
    const c1 = isConsonant(stem, len - 3);
    const v = !isConsonant(stem, len - 2);
    const c2 = isConsonant(stem, len - 1);
    if (c1 && v && c2) {
      return !"wxy".includes(stem[len - 1]);
    }
    return false;
  };

  const replaceSuffix = (word, suffix, replacement, minM = 0, condFn = null) => {
    if (!word.endsWith(suffix)) return null;
    const stem = word.slice(0, word.length - suffix.length);
    if (condFn && !condFn(stem)) return null;
    if (m(stem) < minM && minM > 0) {
      // handled per-rule below with explicit m() checks; kept for clarity
    }
    return stem + replacement;
  };

  let word = w;

  // Step 1a
  if (word.endsWith("sses")) word = word.slice(0, -2);
  else if (word.endsWith("ies")) word = word.slice(0, -2);
  else if (word.endsWith("ss")) { /* no change */ }
  else if (word.endsWith("s")) word = word.slice(0, -1);

  // Step 1b
  let step1bDone = false;
  if (word.endsWith("eed")) {
    const stem = word.slice(0, -3);
    if (m(stem) > 0) word = stem + "ee";
  } else {
    let stem = null;
    if (word.endsWith("ed")) stem = word.slice(0, -2);
    else if (word.endsWith("ing")) stem = word.slice(0, -3);
    if (stem !== null && containsVowel(stem)) {
      word = stem;
      step1bDone = true;
    }
  }
  if (step1bDone) {
    if (word.endsWith("at") || word.endsWith("bl") || word.endsWith("iz")) {
      word = word + "e";
    } else if (endsWithDoubleCons(word) && !/[lsz]$/.test(word)) {
      word = word.slice(0, -1);
    } else if (m(word) === 1 && endsWithCVC(word)) {
      word = word + "e";
    }
  }

  // Step 1c (Porter2/Snowball refinement: only fires when the letter
  // immediately before the trailing Y is itself a consonant, e.g.
  // "happy"->"happi" but NOT "delay"/"spray"/"clayey")
  if (word.length > 1 && word.endsWith("y") &&
      isConsonant(word, word.length - 2) && containsVowel(word.slice(0, -1))) {
    word = word.slice(0, -1) + "i";
  }

  // Step 2
  const step2 = [
    ["ational", "ate"], ["tional", "tion"], ["enci", "ence"], ["anci", "ance"],
    ["izer", "ize"], ["abli", "able"], ["alli", "al"], ["entli", "ent"],
    ["eli", "e"], ["ousli", "ous"], ["ization", "ize"], ["ation", "ate"],
    ["ator", "ate"], ["alism", "al"], ["iveness", "ive"], ["fulness", "ful"],
    ["ousness", "ous"], ["aliti", "al"], ["iviti", "ive"], ["biliti", "ble"],
  ];
  for (const [suf, rep] of step2) {
    if (word.endsWith(suf)) {
      const stem = word.slice(0, word.length - suf.length);
      if (m(stem) > 0) { word = stem + rep; }
      break;
    }
  }

  // Step 3
  const step3 = [
    ["icate", "ic"], ["ative", ""], ["alize", "al"], ["iciti", "ic"],
    ["ical", "ic"], ["ful", ""], ["ness", ""],
  ];
  for (const [suf, rep] of step3) {
    if (word.endsWith(suf)) {
      const stem = word.slice(0, word.length - suf.length);
      if (m(stem) > 0) { word = stem + rep; }
      break;
    }
  }

  // Step 4
  const step4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
  ];
  for (const suf of step4) {
    if (word.endsWith(suf)) {
      let stem = word.slice(0, word.length - suf.length);
      if (suf === "ion") continue;
      if (m(stem) > 1) { word = stem; }
      break;
    }
  }
  // special-case "ion" (must follow s or t)
  if (word.endsWith("ion")) {
    const stem = word.slice(0, -3);
    if (m(stem) > 1 && (stem.endsWith("s") || stem.endsWith("t"))) word = stem;
  }

  // Step 5a
  if (word.endsWith("e")) {
    const stem = word.slice(0, -1);
    if (m(stem) > 1) word = stem;
    else if (m(stem) === 1 && !endsWithCVC(stem)) word = stem;
  }
  // Step 5b
  if (m(word) > 1 && endsWithDoubleCons(word) && word.endsWith("l")) {
    word = word.slice(0, -1);
  }

  return word;
}

