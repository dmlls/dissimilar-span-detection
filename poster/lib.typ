#let size = (
  medium: 40pt,
  normal: 30pt,
  small: 25pt,
  tiny: 20pt,
  more-tiny: 15pt,
  title: 90pt,
  heading-1: 40pt,
  heading-2: 35pt,
  heading-3: 30pt,
)

#let margin = (
  x: 4cm,
  y: 1cm,
)

#let color = (
  primary-blue: rgb(0, 3, 114, 100%),
  primary-blue-80: rgb(0, 3, 114, 80%),
  primary-blue-60: rgb(0, 3, 114, 60%),
  primary-blue-40: rgb(0, 3, 114, 40%),
  primary-blue-20: rgb(0, 3, 114, 20%),
  primary-dark-blue: rgb(15, 23, 42, 100%),
  primary-dark-blue-80: rgb(57, 78, 106, 80%),
  primary-dark-blue-60: rgb(57, 78, 106, 60%),
  primary-dark-blue-40: rgb(57, 78, 106, 40%),
  primary-dark-blue-20: rgb(57, 78, 106, 20%),
  primary-dark-blue-10: rgb(57, 78, 106, 10%),
  primary-dark-blue-5: rgb(57, 78, 106, 5%),
  bg-dark-blue: rgb(15, 23, 42, 100%),
  gray: rgb(229, 231, 235, 100%),
  soft-blue: rgb(242, 247, 254, 100%),
  white: rgb(255, 255, 255, 100%),
  white-80: rgb(255, 255, 255, 80%),
  white-60: rgb(255, 255, 255, 60%),
  white-40: rgb(255, 255, 255, 40%),
  white-20: rgb(255, 255, 255, 20%),
  emphasis: rgb("#0008ff"),
  brown-dark: rgb("#856238"),
  brown: rgb("#b98a51"),
  soft-red: rgb("#ffc8c8"),
  soft-green: rgb("#c8ffc9"),
  white-cover: rgb("#ffffffd8"),
  gray-blue: rgb("#7692b8"),
  gradient-blue-to-dark-blue: gradient.linear(
  rgb("#0d118a"),
  rgb(0, 3, 114, 100%),
    angle: 45deg
  ),
)

#let stroke = 1.2pt + color.primary-blue

#let code(content) = {
  text(font: "JetBrains Mono", content)
}

#let text-box(content, icon: none, text-size: size.normal) = {
  [
    #rect(
      fill: color.soft-blue,
      width: 100%,
      inset: 1.5cm,
      radius: 20pt,
      text(size: text-size, content),
    )
    #if icon != none {
      place(
        top + left,
        dx: -0.3cm,
        dy: 0.2cm,
        image(icon, width: 7%)
      )
    }
  ]
}

#let span(content, similar: false) = {
  highlight(
    fill: if similar {color.soft-green} else {color.soft-red},
  )[{{#content}}]
}

#let reference(number) = {
  text(fill: color.gray-blue)[\[#number\]]
}