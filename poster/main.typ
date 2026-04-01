#import "lib.typ": *
#import "sections.typ": *

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
#set heading(numbering: none, bookmarked: false)

// Set heading margins.
#show heading: set block(above: 1.75em, below: 1em)
#show heading.where(level: 1): set text(size: size.heading-1)
#show heading.where(level: 2): set text(size: size.heading-2)
#show heading.where(level: 3): set text(size: size.heading-3)

#show heading: set text(
  fill: color.gradient-blue-to-dark-blue,
)

// Text setup.
#set par(
  justify: true,
)
#set text(
  lang: "en",
  region: "us",
  font: "Inter",
  size: size.normal,
  hyphenate: true,
  fill: color.primary-blue,
)
#set strong(delta: 200)

// Table setup.
#set table(
  inset: 0.5cm,
  column-gutter: 1em,
  stroke: none,
)
#show table.cell.where(y: 0): set text(weight: "semibold")
#set table.hline(stroke: stroke)

// Do not show supplement in figure captions.
#show figure.caption: set align(left)
#show figure.caption: set text(size: size.more-tiny)

#stack(
  dir: ttb,
  header,
  task,
  diagram,
  v(margin.y),
  align(
    center,
    line(length: 70%, stroke: 2pt + color.soft-blue),
  ),
  block(
    inset: (x: margin.x, y: margin.y),
    sticky: true,
    height: 48%,
    grid(
      row-gutter: 100pt,
      grid(
        columns: (1fr, 1fr),
        grid(
          rows: 2,
          dataset,
          results,
        ),
        {
          grid(
            rows: 2,
            align: horizon,
            row-gutter: 1.7em,
            experiments,
            more-info,
          )
        }
      ),
    ),
  ),
  v(1fr),
  footer,
)
