#import "lib.typ": *

// Set document metadata.
#set document(
  title: [Explainable Semantic Textual Similarity
via Dissimilar Span Detection (LREC 2026)],
  author: ("Diego Miguel Lozano", "Daryna Dementieva", "Alexander Fraser")
)

#set page(
  paper: "a0",
  margin: 0pt,
)

// Heading setup.
#set heading(numbering: none)
// Set heading margins.
#show heading: set block(above: 1.75em, below: 1em)
#show heading.where(level: 1): set text(size: 26pt)
#show heading.where(level: 2): set text(size: 22pt)
#show heading.where(level: 3): set text(size: 18pt)
#show heading.where(level: 4): set text(size: 14pt)

#show heading: set text(
  fill: color.primary-blue,
  size: size.heading-1,
)

// Text setup.
#set text(
  lang: "en",
  region: "us",
  font: "Inter",
  size: size.normal,
  hyphenate: true,
  fill: color.primary-dark-blue,
)

// Make figure captions adapt to the width of the image.
#show figure: fig => {
  if fig.body.has("width") {
    let fig-width = fig.body.width;
    show figure.caption: box.with(width: fig-width)
    fig
  } else {
    fig
  }
}

// Do not show supplement in figure captions.
#show figure.caption: it => context [
  #text(size: 10.5pt)[#it.body]
]

#show table.cell.where(y: 0): set text(white, weight: "bold")

// Underline links and references.
#show link: underline
#show ref: underline

#stack(
  dir: ttb,
  header(),
  block(
    inset: (x: 150pt, y: 80pt),
    sticky: true,
    grid(
      rows: 2,
      row-gutter: 100pt,
      image("img/dissimilar-span-detection.png"),
      grid(
        columns: (1fr, 1fr),
        grid(
          rows: 2,
          row-gutter: 100pt,
          dataset(),
          results(),
        ),
        experiments(),
      ),
    ),
  ),
  v(1fr),
  footer(),
)