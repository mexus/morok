import fs from 'node:fs';
import path from 'node:path';

const REPO_URL = 'https://github.com/npatsakula/morok/tree/main';

export default function readmeIntroPlugin(context) {
  return {
    name: 'readme-intro',
    async loadContent() {
      let readme = fs.readFileSync(
        path.resolve(context.siteDir, '..', 'README.md'),
        'utf-8'
      );

      // Rewrite relative markdown links to absolute GitHub URLs so
      // Docusaurus doesn't treat them as broken doc references.
      readme = readme.replace(
        /\]\((?!https?:\/\/)([^)]+)\)/g,
        (match, relPath) => `](${REPO_URL}/${relPath})`,
      );

      const frontmatter = '---\nsidebar_label: Introduction\n---\n\n';
      const outPath = path.resolve(context.siteDir, 'docs', 'introduction.md');
      fs.writeFileSync(outPath, frontmatter + readme);
    },
  };
}
