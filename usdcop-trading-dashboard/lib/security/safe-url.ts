/** Return only absolute HTTP(S) URLs from untrusted feed data. */
export function safeExternalUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null;
  try { const u = new URL(value); return u.protocol === 'http:' || u.protocol === 'https:' ? u.href : null; }
  catch { return null; }
}
