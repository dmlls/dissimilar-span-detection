#import "@preview/polylux:0.4.0": *
#import "utils.typ": *

#set document(
  title: [Embedding-DSD],
  author: "Diego Miguel Lozano"
)

#let text-font-size = 13pt
#let heading-1-font-size = 28pt
#let heading-2-font-size = 28pt

#set page(paper: "presentation-16-9", margin: (bottom: 2.8cm, rest: 2cm))
#set heading(outlined: false, bookmarked: false)
#set text(size: text-font-size, font: "Inter", fill: color.primary-dark-blue)
#show heading.where(level: 1): set text(
  size: heading-1-font-size,
  fill: color.primary-blue,
)
#show heading.where(level: 2): set text(
  size: heading-2-font-size,
  fill: color.brown
)
#show figure: set text(size: text-font-size - 7pt)
#show figure: fig => {
  let fig-width = measure(fig.body).width
  // Would be better to use box.width, but that doesn't work.
  show figure.caption: rect.with(width: fig-width, stroke: none)
  set align(center)
  fig
}
#set list(indent: 1em)
#set strong(delta: 200)
#set highlight(extent: 4pt, radius: 5pt)
#show highlight: it => {
  h(4pt)
  it
  h(4pt)
}

#set page(
  background: image("img/bg-page.png", width: 100%),
)

#let code(content) = {
  text(font: "JetBrains Mono", content)
}

#let rectangle(content, width: 10cm) = {
  set align(center)
  rect(
    width: width,
    stroke: none,
    fill: color.soft-blue,
    outset: 0.4em,
    radius: 5pt,
    code(content),
  )
}

#let ngram(content) = {
  highlight(
    fill: white,
    stroke: 0.05em + color.primary-blue-20,
    extent: 0.3em,
    radius: 0.3em,
    text(fill: color.emphasis, weight: "semibold", content)
  )
}

#slide[
  = Embedding-DSD
  #set align(center + horizon)

  #stack(dir: ltr, spacing: 2em,
    [Sentence 1:],
    rectangle[the bird flies fast over the hill]
  )
  #v(0.2cm)
  #stack(dir: ltr, spacing: 2em,
    [Sentence 2:],
    rectangle[the car rides fast over the hill]
  )
]

#slide[
  = Embedding-DSD
  #set align(center + horizon)

  #stack(dir: ltr, spacing: 2em,
    [Sentence 1:],
    rectangle[the bird flies fast over the hill]
  )
  #v(0.2cm)
  #stack(dir: ltr, spacing: 2em,
    [Sentence 2:],
    rectangle[the car rides fast over the hill]
  )

  #v(0.5cm)
  #text(
    size: text-font-size + 3pt,
  )[*Base Similarity: 0.64*]
]

#slide[
  = Embedding-DSD
  #set align(center + horizon)
  #set text(size: text-font-size - 3pt)

  #place(
    top + right,
    [
      #set align(horizon)
      #stack(dir: ltr, spacing: 2em,
        [Sentence 1:],
        rectangle(width: 8cm)[the #ngram[bird] flies fast over the hill]
      )
      #v(0.2cm)
      #stack(dir: ltr, spacing: 2em,
        [Sentence 2:],
        rectangle(width: 8cm)[the car rides fast over the hill]
      )

      #v(0.1cm)
      #text(
        size: text-font-size - 1pt,
      )[*Base Similarity: 0.64*]
    ]
  )

  #v(2cm)

  #set text(size: text-font-size - 3pt)
  #show: later

  #rectangle(width: 8cm)[the car rides fast over the hill]

  #v(2em)

  #uncover((beginning: 3))[
    #grid(
      columns: (10cm, 10cm),
      column-gutter: 1.5em,
      align: (right, left),
      rectangle(width: 8cm)[#ngram[bird] car rides fast over the hill],
      text(
        font: "JetBrains Mono",
        one-by-one[STS: 0.77][#h(0.7em) | #h(0.7em) 0.77 - 0.64 => #[#sym.Delta]G = 0.13]
      )
    )
  ]
  #uncover((beginning: 4))[
    #grid(
      columns: (10cm, 10cm),
      column-gutter: 1.5em,
      align: (right, left),
      rectangle(width: 8cm)[the #ngram[bird] rides fast over the hill],
      text(
        font: "JetBrains Mono",
        one-by-one[STS: 0.93][#h(0.7em) | #h(0.7em) 0.93 - 0.64 => #[#sym.Delta]G = 0.29]
      )
    )
  ]
  #uncover((beginning: 5))[
    #grid(
      columns: (10cm, 10cm),
      column-gutter: 1.5em,
      align: (right, left),
      rectangle(width: 8cm)[the car #ngram[bird] fast over the hill],
      text(
        font: "JetBrains Mono",
        one-by-one[STS: 0.82][#h(0.7em) | #h(0.7em) 0.82 - 0.64 => #[#sym.Delta]G = 0.18]
      )
    )
  ]
  #uncover((beginning: 6))[
    #grid(
      columns: (10cm, 10cm),
      column-gutter: 1.5em,
      align: (right, left),
      rectangle(width: 8cm)[the car rides #ngram[bird] over the hill],
      text(
        font: "JetBrains Mono",
        one-by-one[STS: 0.74][#h(0.7em) | #h(0.7em) 0.74 - 0.64 => #[#sym.Delta]G = 0.10]
      )
    )
  ]
  #uncover((beginning: 7))[
    #grid(
      columns: (10cm, 10cm),
      column-gutter: 1.5em,
      align: (right, left),
      rectangle(width: 8cm)[the car rides fast #ngram[bird] the hill],
      text(
        font: "JetBrains Mono",
        one-by-one[STS: 0.60][#h(0.7em) | #h(0.7em) 0.60 - 0.64 => #[#sym.Delta]G = -0.04]
      )
    )
  ]
  #uncover((beginning: 8))[
    #grid(
      columns: (10cm, 10cm),
      column-gutter: 1.5em,
      align: (right, left),
      rectangle(width: 8cm)[the car rides fast over #ngram[bird] hill],
      text(
        font: "JetBrains Mono",
        one-by-one[STS: 0.73][#h(0.7em) | #h(0.7em) 0.73 - 0.64 => #[#sym.Delta]G = 0.09]
      )
    )
  ]
  #uncover((beginning: 9))[
    #grid(
      columns: (10cm, 10cm),
      column-gutter: 1.5em,
      align: (right, left),
      rectangle(width: 8cm)[the car rides fast over the #ngram[bird]],
      text(
        font: "JetBrains Mono",
        one-by-one[STS: 0.73][#h(0.7em) | #h(0.7em) 0.64 - 0.64 => #[#sym.Delta]G = 0.64]
      )
    )
  ]
]


#slide[
  = Embedding-DSD
  #set align(center + horizon)
  #set text(size: text-font-size - 3pt)

  #place(
    top + right,
    [
      #set align(horizon)
      #stack(dir: ltr, spacing: 2em,
        [Sentence 1:],
        rectangle(width: 8cm)[the bird flies fast over the hill]
      )
      #v(0.2cm)
      #stack(dir: ltr, spacing: 2em,
        [Sentence 2:],
        rectangle(width: 8cm)[the car rides fast over the hill]
      )

      #v(0.1cm)
      #text(
        size: text-font-size - 1pt,
      )[*Base Similarity: 0.64*]
    ]
  )

  #v(2cm)

  #set text(size: text-font-size - 3pt)

  #rectangle(width: 8cm)[the #text(fill: color.primary-blue)[*car rides*] fast over the hill]

  #v(2em)

  #grid(
    columns: (10cm, 10cm),
    column-gutter: 1.5em,
    align: (right, left),
    rectangle(width: 8cm)[#ngram[bird] car rides fast over the hill],
    text(
      font: "JetBrains Mono",
      [STS: 0.77 #h(0.7em) | #h(0.7em) 0.77 - 0.64 => #[#sym.Delta]G = 0.13]
    )
  )
  #grid(
    columns: (10cm, 10cm),
    column-gutter: 1.5em,
    align: (right, left),
    rectangle(width: 8cm)[the #ngram[bird] rides fast over the hill],
    text(
      font: "JetBrains Mono",
      [STS: 0.93 #h(0.7em) | #h(0.7em) 0.93 - 0.64 => #text(weight: "bold", fill: color.primary-blue)[#[#sym.Delta]G = 0.29]]
    )
  )
  #grid(
    columns: (10cm, 10cm),
    column-gutter: 1.5em,
    align: (right, left),
    rectangle(width: 8cm)[the car #ngram[bird] fast over the hill],
    text(
      font: "JetBrains Mono",
      [STS: 0.82 #h(0.7em) | #h(0.7em) 0.82 - 0.64 =>  #text(weight: "bold", fill: color.primary-blue)[#[#sym.Delta]G = 0.18]]
    )
  )
  #grid(
    columns: (10cm, 10cm),
    column-gutter: 1.5em,
    align: (right, left),
    rectangle(width: 8cm)[the car rides #ngram[bird] over the hill],
    text(
      font: "JetBrains Mono",
      [STS: 0.74 #h(0.7em) | #h(0.7em) 0.74 - 0.64 => #[#sym.Delta]G = 0.10]
    )
  )
  #grid(
    columns: (10cm, 10cm),
    column-gutter: 1.5em,
    align: (right, left),
    rectangle(width: 8cm)[the car rides fast #ngram[bird] the hill],
    text(
      font: "JetBrains Mono",
      [STS: 0.60 #h(0.7em) | #h(0.7em) 0.60 - 0.64 => #[#sym.Delta]G = -0.04]
    )
  )
  #grid(
    columns: (10cm, 10cm),
    column-gutter: 1.5em,
    align: (right, left),
    rectangle(width: 8cm)[the car rides fast over #ngram[bird] hill],
    text(
      font: "JetBrains Mono",
      [STS: 0.73 #h(0.7em) | #h(0.7em) 0.73 - 0.64 => #[#sym.Delta]G = 0.09]
    )
  )
  #grid(
    columns: (10cm, 10cm),
    column-gutter: 1.5em,
    align: (right, left),
    rectangle(width: 8cm)[the car rides fast over the #ngram[bird]],
    text(
      font: "JetBrains Mono",
      [STS: 0.73 #h(0.7em) | #h(0.7em) 0.64 - 0.64 => #[#sym.Delta]G = 0.64]
    )
  )
]

#slide[
  = Embedding-DSD
  #set align(center + horizon)
  #set text(size: text-font-size - 3pt)

  #place(
    top + right,
    [
      #set align(horizon)
      #stack(dir: ltr, spacing: 2em,
        [Sentence 1:],
        rectangle(width: 8cm)[#ngram[the bird] flies fast over the hill]
      )
      #v(0.2cm)
      #stack(dir: ltr, spacing: 2em,
        [Sentence 2:],
        rectangle(width: 8cm)[the car rides fast over the hill]
      )

      #v(0.1cm)
      #text(
        size: text-font-size - 1pt,
      )[*Base Similarity: 0.64*]
    ]
  )

  #v(2cm)

  #set text(size: text-font-size - 3pt)
  #show: later

  #rectangle(width: 8cm)[the car rides fast over the hill]

  #v(2em)

  #uncover((beginning: 3))[
    #grid(
      columns: (10cm, 10cm),
      column-gutter: 1.5em,
      align: (right, left),
      rectangle(width: 8cm)[#ngram[the bird] rides fast over the hill],
      text(
        font: "JetBrains Mono",
        one-by-one[STS: 0.93][#h(0.7em) | #h(0.7em) 0.93 - 0.64 => #[#sym.Delta]G = 0.29]
      )
    )
  ]
  #uncover((beginning: 4))[
    #grid(
      columns: (10cm, 10cm),
      column-gutter: 1.5em,
      align: (right, left),
      rectangle(width: 8cm)[the #ngram[the bird] fast over the hill],
      text(
        font: "JetBrains Mono",
        one-by-one[STS: 0.82][#h(0.7em) | #h(0.7em) 0.82 - 0.64 => #[#sym.Delta]G = 0.18]
      )
    )
  ]
  #uncover((beginning: 5))[
    #text(size: 20pt)[...]
  ]
]

#slide[
  = Embedding-DSD
  #set align(center + horizon)

  #ngram[the] -> #code[\[0.13, 0.29, 0.09, 0.10, ..., 0.02\]]

  #show: later
  #ngram[car] -> #code[\[0.29, 0.29, 0.18, 0.15, ..., 0.09\]]

  #show: later
  #ngram[rides] -> #code[\[0.18, 0.24, 0.17, 0.12, ..., 0.08\]]

  #show: later
  ...
]

#slide[
  = Embedding-DSD
  #set align(center + horizon)

  #set text(size: text-font-size + 4pt)
  #rect(
    stroke: none,
    fill: color.soft-blue,
    outset: 1em,
    radius: 6pt
  )[
    $"AggrGain"_"unigram" = 1 / n dot sum_(i=1)^n "gains"_i / i $
  ]
]
