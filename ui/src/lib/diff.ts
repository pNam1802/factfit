/** Word-level diff between an original bullet and its rewrite (longest common subsequence). */

export type Token = { word: string; kind: "same" | "added" | "removed" };

function words(text: string): string[] {
  return text.split(/\s+/).filter(Boolean);
}

/** Compare words loosely: case and trailing punctuation do not count as a change. */
function key(word: string): string {
  return word.toLowerCase().replace(/[.,;:!?]+$/, "");
}

/** Tokens to show for the original (same / removed) and for the rewrite (same / added). */
export function wordDiff(original: string, rewrite: string): { before: Token[]; after: Token[] } {
  const a = words(original);
  const b = words(rewrite);
  // lcs[i][j] = length of the longest common subsequence of a[i:] and b[j:]
  const lcs: number[][] = Array.from({ length: a.length + 1 }, () => Array(b.length + 1).fill(0));
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lcs[i][j] =
        key(a[i]) === key(b[j]) ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const before: Token[] = [];
  const after: Token[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (key(a[i]) === key(b[j])) {
      before.push({ word: a[i++], kind: "same" });
      after.push({ word: b[j++], kind: "same" });
    } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
      before.push({ word: a[i++], kind: "removed" });
    } else {
      after.push({ word: b[j++], kind: "added" });
    }
  }
  while (i < a.length) before.push({ word: a[i++], kind: "removed" });
  while (j < b.length) after.push({ word: b[j++], kind: "added" });
  return { before, after };
}
