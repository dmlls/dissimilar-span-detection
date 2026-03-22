#let size = (
  normal: 40pt,
  small: 20pt,
  title: 90pt,
  heading-1: 40pt,
)

#let color = (
  primary-blue: rgb(0, 3, 114, 100%),
  primary-dark-blue: rgb(57, 78, 106, 100%),
  primary-dark-blue-80: rgb(57, 78, 106, 80%),
  primary-dark-blue-60: rgb(57, 78, 106, 60%),
  primary-dark-blue-40: rgb(57, 78, 106, 40%),
  primary-dark-blue-20: rgb(57, 78, 106, 20%),
  primary-dark-blue-10: rgb(57, 78, 106, 10%),
  primary-dark-blue-5: rgb(57, 78, 106, 5%),
  bg-dark-blue: rgb(15, 23, 42, 100%),
  gray: rgb(229, 231, 235, 100%),
  soft-gray: rgb(242, 247, 254, 100%),
  white: rgb(255, 255, 255, 100%),
  white-80: rgb(255, 255, 255, 80%),
  white-60: rgb(255, 255, 255, 60%),
  white-40: rgb(255, 255, 255, 40%),
  white-20: rgb(255, 255, 255, 20%),
  brown-dark: rgb("#856238"),
  brown: rgb("#b98a51"),
  soft-red: rgb("#ffc8c8"),
  soft-green: rgb("#c8ffc9"),
  white-cover: rgb("#ffffffd8"),
  gradient-blue-to-dark-blue: gradient.linear(
    rgb(33, 9, 209, 100%),
    rgb(18, 6, 111, 100%),
    angle: 45deg
  ),
)


#let header() = {
  block(
    width: 100%,
    fill: tiling(
      image("img/bg_blur.png")
    ),
    align(
      center,
      block(
        inset: (top: 100pt, bottom: 50pt, y: 120pt),
        stack(
          dir: ttb,
        spacing: 60pt,
          rect(
            fill: color.white-60,
            radius: 20pt,
            stroke: (3pt + color.white),
            inset: (x: 40pt, y: 30pt),
            text(
              font: "DIN Pro",
              stretch: 75%,
              weight: "medium",
              size: size.title - 30pt,
            )[LREC 2026],
          ),
          stack(
            dir: ttb,
            spacing: 40pt,
            text(size: size.title, weight: "black")[Explainable Semantic Textual Similarity \ via Dissimilar Span Detection],
            v(60pt),
            [
              Diego Miguel Lozano #super[1 , †],
              Daryna Dementieva #super[1, 2],
              Alexander Fraser #super[1, 2 ]
            ],
            text(size: size.small)[
              #super[1] School of Computation, Information and Technology, Technical University of Munich (TUM) \
              #super[2] Munich Center for Machine Learning (MCML) \

              #super[†] Currently affiliated to ELLIS Alicante.
            ]

          )
        )
      )
    )
  )
}

#let footer() = {
  set text(fill: color.gray)
  block(
    width: 100%,
    fill: color.bg-dark-blue,
    align(
      center,
      block(
        inset: (x: 100pt, y: 80pt),
        stack(
          dir: ltr,
          spacing: 100pt,
          image("img/logo_tum.svg", height: 70pt),
          image("img/logo_mcml.png", height: 70pt),
        )
      )
    )
  )
}

#let dataset() = {
  rect(
    fill: color.white,
    width: 100%,
    inset: 60pt,
    radius: 20pt,
    heading(level: 1)[1. Span Similarity Dataset (SSD)],
  )
}

#let experiments() = {
  rect(
    fill: color.soft-gray,
    width: 100%,
    height: 40%,
    inset: 60pt,
    radius: 20pt,
    heading(level: 1)[2. Experimental Setup],
  )
}

#let results() = {
  rect(
    fill: color.white,
    width: 100%,
    inset: 60pt,
    radius: 20pt,
    heading(level: 1)[3. Results],
  )
}
