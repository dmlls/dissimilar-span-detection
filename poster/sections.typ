#import "lib.typ": *

#let header = {
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
          stack(
            dir: ttb,
            spacing: 40pt,
            text(size: size.title, fill: color.gradient-blue-to-dark-blue, weight: "black")[Explainable Semantic Textual Similarity \ via Dissimilar Span Detection],
            v(60pt),
            text(size: size.medium)[
              #link(
                "https://www.diegomiguel.me/",
                "Diego Miguel Lozano"
              ) #super[1 , †],
              #link(
                "https://dardem.github.io/",
                "Daryna Dementieva"
              ) #super[1, 2],
              #link(
                "https://alexfraser.github.io/",
                "Alexander Fraser"
              )
              #super[1, 2 ]
            ],
            text(size: size.tiny)[
              #super[1] School of Computation, Information and Technology, Technical University of Munich (TUM) \
              #super[2] Munich Center for Machine Learning (MCML) \

              #super[†] Currently affiliated to ELLIS Alicante
            ]
          )
        )
      )
    )
  )
}

#let footer = {
  set text(fill: color.gray)
  block(
    width: 100%,
    fill: color.bg-dark-blue,
    grid(
      columns: (1fr, 1fr, 1fr),
      align: (left, center, right).map(it => it + horizon),
      inset: 1.7em,
      place(horizon)[
        #set par(spacing: 1.8em)
        #set text(fill: color.white, size: size.more-tiny - 8pt)

        #text(size: size.more-tiny - 6pt)[*References:*]

        \[1\] Marco Tulio Ribeiro et al. 2016. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. In _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_, pages 1135-1144.

        \[2\] Scott M Lundberg and Su-In Lee. 2017. A Unified Approach to Interpreting Model Predictions. In _Advances in Neural Information Processing Systems 30_, pages 4765-4774.
      ],
      stack(
        dir: ltr,
        spacing: 100pt,
        image("img/logo_tum.svg", height: 47pt),
        image("img/logo_mcml.png", height: 50pt),
      ),
      place(right + horizon)[
        #set align(center)
        #set par(spacing: 0.3em)
        #text(
          font: "DIN Pro",
          stretch: 75%,
          weight: "medium",
          size: size.title - 40pt,
        )[LREC 2026]

        #text(size: size.small - 2pt)[Palma de Mallorca]
      ]
    )
  )
}

#let task = {
  block(
    inset: (x: margin.x, y: margin.y), width: 100%,
  )[
    #set align(center)
    #text-box(width: 66.7cm, icon: "img/alert-triangle.svg")[
      *Cosine similarity is not always enough!*

      #set text(size.small)
      Outputting a single, non-interpretable number can mask fundamental differences between the texts being compared. \
      We introduce the task of *Dissimilar Span Detection (DSD)*: Given two texts, identifying spans pairs with a common semantic function, but differing meanings.
    ]
  ]
}

#let diagram = {
  set align(center)
  block(
    width: 100%,
    inset: (x: margin.x, top: -1cm),
  )[
    #image("img/dissimilar-span-detection.pdf", width: 66.7cm)
  ]
}

#let dataset = {
  block(
    width: 100%,
    inset: (bottom: 15pt, top: 10pt, rest: 60pt),
  )[
    #heading(level: 1)[1. Span Similarity Dataset (SSD)]

    #set text(size: size.small)
    #show table.cell: set text(size: size.tiny)
    #show table.cell: it => {
      if it.x >= 2 and it.y > 0 {
        set text(font: "JetBrains Mono")
        highlight(fill: color.soft-blue, extent: 5pt)[#code(it)]
      } else {
        it
      }
    }

    New dataset (1,000 samples) crafted specifically for the task of DSD:

    #figure(
      table(
        columns: (1fr, 1fr, 0.5fr, 0.5fr),
        align: (left, left, center, center).map(it => it + horizon),
        table.header(
          table.hline(),
          [Sentence 1], [Sentence 2], [Span \ Similarity], [Sentence \ Similarity],
          table.hline(),
        ),
        [It was #span[restored] in the #span(similar: true)[1980s].],
        [It was #span[destroyed] in the #span(similar: true)[eighties].],
        [0,1], [0],
        [Thank you for #span(similar: true)[wasting my money].],
        [Thank you for #span(similar: true)[misusing my funds].],
        [1], [1],
        [There are depots at #span[Quilpie] and #span[Roma].],
        [There are depots at #span[Brisbane] and #span[Sydney].],
        [0,0], [0],
        table.hline(),
      ),
      caption: [Examples of shorter sentences from the SSD. Spans are denoted using double curly braces, and then annotated respectively with a 0, in case the span pair differs in meaning, or 1, if they are equivalent. We also annotate the sentence similarity of the pair as a whole.]
    )

    Created in a semi-automatic way in two steps: (1) We let an LLM replace the spans, and (2) we manually review them and annotate the similarity. To ensure the correctness of the spans, six annotators manually annotate spans in 100 samples, achieving good inter-annotator and annotator-dataset agreement ($kappa in [0.61, 0.91] $).
  ]
}

#let experiments = {
  rect(
    fill: color.soft-blue,
    width: 100%,
    inset: 60pt,
    radius: 20pt,
    block[
      #heading(level: 1)[2. Experimental Setup]

      #set text(size: size.small)
      #set par(spacing: 1em)
      #let method(name, desc, extended-desc: none, column-size: 5cm) = {
        block(
          below: 0.9em,
          rect(
            width: 100%,
            stroke: none,
            fill: color.white,
            radius: 5pt,
            inset: (x: 1em, y: 0.8em),
            [
              #align(
                horizon,
                grid(
                  columns: (column-size, auto),
                  column-gutter: 2em,
                  align: (left, left),
                  strong(name),
                  text(size: size.tiny, desc),
                )
              )
              #if extended-desc != none {
                line(length: 100%, stroke: 1.2pt + color.soft-blue)
                text(size: size.tiny, extended-desc)
              }
            ]
          )
        )
      }

      We propose *5 methods* to tackle the task of DSD:

      #method(
        [SHAP-DSD#super[\*]],
        [
          Based on the SHAP framework #reference[1]. We calculate the dissimilarity score as:

          #rect(
            stroke: none,
            fill: color.soft-blue,
            outset: 0.4em,
            radius: 5pt
          )[
            $"DissimilarityScore"(a, b) = 1 - "CosineSimilarity"(a, b)$
          ]

          Tokens with dissimilarity above a threshold are considered as dissimilar.
        ]
      )
      #method(
        [LIME-DSD#super[\*]],
        [
          Similar to SHAP-DSD, but with the explanation weights predicted by LIME #reference[2].
        ]
      )
      #method(
        [LLM-DSD],
        [
          Uses an LLM to detect the dissimilar spans.
        ]
      )
      #method(
        column-size: auto,
        [Token-Classification-DSD],
        [
          A model is fine-tuned to predict dissimilar spans.
        ]
      )
      #method(
        column-size: auto,
        [Embedding-DSD#super[\*]],
        [
          This method constitutes a *novel contribution*.
        ],
        extended-desc: [
          We calculate all possible unigram, bigram, #sym.dots, _n_-gram replacements from the first sentence to the second. For example, for the following 2 sentences:

          #align(
            center,
            block(spacing: 2em)[
              #set text(size: size.tiny - 3pt)
              #let rectangle(content) = {
                rect(
                  width: 12cm,
                  stroke: none,
                  fill: color.soft-blue,
                  outset: 0.4em,
                  radius: 5pt,
                  code(content),
                )
              }
              #rectangle[the bird flies fast over the hill]
              #v(0.2cm)
              #rectangle[the car rides fast over the hill]
            ]
          )

          If we were considering the trigram #text(size: size.tiny - 2pt, code[\['the', 'bird', 'flies'\]]), we would get the replacements:

          #align(
            center,
            block(spacing: 2em)[
              #set text(size: size.tiny - 3pt)
              #stack(
                dir: ltr,
                spacing: 0.6em,
                rect(
                  width: auto,
                  stroke: none,
                  fill: color.gray-blue,
                  outset: 0.4em,
                  radius: 5pt,
                  ("1", "2", "3", "4", "5").map(code).map(
                    it => text(
                      fill: color.white,
                      it)
                  ).join([ \ ])
                ),
                rect(
                  stroke: none,
                  fill: color.soft-blue,
                  inset: (right: 0.8em),
                  outset: 0.4em,
                  radius: (right: 5pt),
                )[
                  #set align(left)
                  #let ngram(content) = {
                    highlight(
                      fill: color.white,
                      extent: 0.3em,
                      text(fill: color.emphasis, content)
                    )
                  }
                  #(
                    [#ngram[the bird flies] fast over the hill],
                    [the #ngram[the bird flies] over the hill],
                    [the car #ngram[the bird flies] the hill],
                    [the car rides #ngram[the bird flies] hill],
                    [the car rides fast #ngram[the bird flies]],
                   ).map(code).join([ \ ])
                ]
              )
            ]
          )
          We then calculate the _similarity gain_ for each _n_-gram by comparing the resulting cosine similarity to the _base similarity_ (i.e., the cosine similarity between the original sentences). Each unigram is assigned the gains coming from each of the _n_-grams in which it appears. These are then combined through an _aggregation function_:

          #align(
            center,
            block(spacing: 2em)[
              #rect(
                stroke: none,
                fill: color.soft-blue,
                outset: 0.4em,
                radius: 5pt
              )[
                $"AggrGain"_"unigram" = 1 / n dot sum_(i=1)^n "gains"_i / i $
              ]
            ]
          )

          Unigrams with a gain above a certain threshold are considered as dissimilar.
        ]
      )
      #place(
        bottom,
        dy: 1.2cm,
        text(size: size.more-tiny)[\* Methods that require a threshold. We find the optimal threshold by evaluating on the validation split of the dataset.]
      )
    ]
  )
}

#let results = {
  block(
    height: 100%,
    width: 100%,
    inset: (bottom: 0pt, top: 70pt, rest: 60pt),
  )[
    #set text(size: size.small)
    #heading(level: 1)[3. Results]

    #let model(content) = {
      set text(size: size.more-tiny - 3pt)
      code(content)
    }

    #set table(column-gutter: 0.6em, row-gutter: -0.15em)
    #[
      #show table.cell: set text(size: size.more-tiny)
      #show table.cell.where(y: 1): set text(size: size.more-tiny, weight: "semibold")
      #figure(
        table(
          columns: (0.77fr, 0.49fr, 0.48fr, 0.47fr, 0.36fr, 0.4fr, 0.4fr, 0.4fr),
          align: (left, center, center, center, center, center, center, center).map(it => it + horizon),
          table.header(
            table.hline(),
            [], [],
            table.hline(start: 2, end: 6, stroke: stroke),
            table.cell(colspan: 4)[SSD],
            table.hline(start: 6, end: 8, stroke: stroke),
            table.cell(colspan: 2)[SemEval-2016],
            [Method], [Model Size], [F1-Global], [F1-NoDiff],
            [F1-Diff], [Time], [F1-Diff], [Time],
            table.hline(),
          ),
          [LIME \ #model[all-mpnet-base-v2]], [109M], [0.463], [0.782], [0.223], [1981.81], [0.109], [199.89],
          [SHAP \ #model[all-MiniLM-L6-v2]], [22.7M], [0.366], [0.434], [0.131], [44.47], [0.306], [4.52],
          [Embedding \ #model[all-mpnet-base-v2]], [109M], [0.469], [0.529], [0.423], [11.28], [0.352], [1.13],
          [Embedding \ #model[text-embedding-004]], [Billions], [0.547], [0.666], [0.459], [611.42], [0.338], [34.57],
          [LLM \ #model[Claude 3.5 Sonnet]], [Billions], [*0.750*], [*0.895*], [*0.640*], [337.80], [0.389], [39.22],
          [Token Classification \ #model[roberta-base]], [125M], [0.690], [0.838], [0.574], [*1.23*], [*0.441*], [*0.10*],
          table.hline(),
        ),
        caption: [Summary of the results. We adopt a 5-fold cross-validation scheme. To account for false positives, we report separate F1 scores for those sentence-pairs that contain no dissimilar spans (_NoDiff_), and those that do (_Diff_). Evaluation times are reported in minutes.]
      )
    ]

    #v(0.5cm)
    #heading(level: 2)[3.1 #h(0.5cm) Downstream Task -- Paraphrase Detection]

    #show table.cell: set text(size: size.tiny - 2pt)
    #grid(
      columns: (1fr, 1fr),
      column-gutter: 2.3em,
      align: horizon,
      figure(
        table(
          columns: (1fr, 0.55fr, 0.65fr),
          align: (left, center, center).map(it => it + horizon),
          table.header(
            table.hline(),
            [Model], [STS], [STS + DSD],
            table.hline(),
          ),
          text(size: size.tiny - 5pt, code[all-MiniLM-L6-v2]), [0.720], [*0.808*],
          text(size: size.tiny - 5pt, code[all-mpnet-base-v2]), [0.795], [*0.868*],
          table.hline(),
        ),
        caption: [Comparison of accuracies on the PAWS-Wiki Labeled using uniquely STS or combining STS with DSD.]
      ),
      text-box(
        icon: "img/check-mark.svg",
        icon-size: 7%,
        text-size: size.small + 0.7pt,
      )[DSD improves performance on the task with no fine-tuning.]
    )
  ]
}

#let more-info = {
  set align(center + horizon)
  block(
    width: 60%,
    grid(
      columns: (auto, auto),
      column-gutter: 1em,
      align: (right, center),
      stack(
        dir: ttb,
        spacing: 1.5em,
        [*Want to know more?*],
        place(
          right,
          dx: 1.5cm,
          dy: -1.2cm,
          rotate(5deg, image("img/arrow.svg", height: 1cm)),
        )
      ),
      stack(
        dir: ttb,
        spacing: 0.8em,
        v(-0.3cm),
        image("img/qr.svg", height: 8%),
        link(
          "https://dmlls.github.io/dissimilar-span-detection/",
          text(size: size.more-tiny)[
            #box(
              height: 1em,
              baseline: 15%,
              image("img/link.svg")
            )
            dmlls.github.io/ \ #v(-0.4cm) dissimilar-span-detection
          ],
        )
      )
    )
  )
}
