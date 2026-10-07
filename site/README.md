# Project Page

- `index.html`: paper overview, qualitative comparisons, and paper result tables.
- `styles.css`: desktop and mobile layout.
- `app.js`: publication links, example switching, image enlargement, and BibTeX copying.
- `assets/`: the paper’s four qualitative examples under six settings, two method figures, local fonts, Font Awesome / Academicons icons, and the [Hugging Face logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg).

## Preview

From the repository root:

```bash
python3 -m http.server 8080 --directory site
```

Open [the local preview](http://localhost:8080).

## Publish

The GitHub Pages workflow publishes only `site/`. In the repository’s **Settings → Pages**, select **GitHub Actions** as the source, then run the **Project page** workflow. Later changes to `site/` on `main` deploy automatically.

Paper and arXiv URLs are defined in the `links` object in `app.js`. Hugging Face and Code links are in `index.html`.
