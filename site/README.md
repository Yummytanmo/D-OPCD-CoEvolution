# Project Page

- `index.html`: paper overview, qualitative comparisons, and paper result tables.
- `styles.css`: desktop and mobile layout.
- `app.js`: publication links, example switching, image enlargement, and BibTeX copying.
- `assets/`: the paper’s four qualitative examples under six settings, two method figures, local fonts, and Font Awesome / Academicons button icons.

## Preview

From the repository root:

```bash
python3 -m http.server 8080 --directory site
```

Open [the local preview](http://localhost:8080).

## Publish

The GitHub Pages workflow publishes only `site/`. In the repository’s **Settings → Pages**, select **GitHub Actions** as the source, then run the **Project page** workflow. Later changes to `site/` on `main` deploy automatically.

Publication URLs are defined in the `links` object in `app.js`. Paper opens the arXiv PDF; arXiv opens the abstract page.
