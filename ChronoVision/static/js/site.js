const scrollButton = document.querySelector('.scroll-to-top');
const copyButton = document.querySelector('#copy-bibtex');
const bibtexCode = document.querySelector('#bibtex-code');

scrollButton?.addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

window.addEventListener('scroll', () => {
  scrollButton?.classList.toggle('visible', window.scrollY > 300);
}, { passive: true });

copyButton?.addEventListener('click', async () => {
  const text = bibtexCode?.textContent?.trim();
  if (!text) return;

  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
  }

  const label = copyButton.querySelector('.copy-text');
  copyButton.classList.add('copied');
  if (label) label.textContent = 'Copied';

  window.setTimeout(() => {
    copyButton.classList.remove('copied');
    if (label) label.textContent = 'Copy';
  }, 1800);
});
