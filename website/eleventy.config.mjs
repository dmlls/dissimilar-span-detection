import { eleventyImageTransformPlugin } from "@11ty/eleventy-img";
import { RenderPlugin } from "@11ty/eleventy";
import fs from "fs";
import path from "path";
import postcss from "postcss";
import tailwindcss from "@tailwindcss/postcss";

export default function (eleventyConfig) {
  eleventyConfig.on("eleventy.before", async () => {
    const tailwindInputPath = path.resolve("./styles/app.css");
    const tailwindOutputPath = "./_site/styles/app.css";
    const cssContent = fs.readFileSync(tailwindInputPath, "utf8");
    const outputDir = path.dirname(tailwindOutputPath);

    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const result = await postcss([tailwindcss()]).process(cssContent, {
      from: tailwindInputPath,
      to: tailwindOutputPath,
    });

    fs.writeFileSync(tailwindOutputPath, result.css);
  });

  eleventyConfig.addPassthroughCopy("assets/");
  eleventyConfig.addPassthroughCopy("scripts/");
  eleventyConfig.addPassthroughCopy("styles/fonts");

  // Image plugin
  eleventyConfig.addPlugin(eleventyImageTransformPlugin, {
    formats: ["webp", "jpeg", "png", "svg"],
    urlPath: "img/",
    defaultAttributes: {
      loading: "lazy",
      decoding: "async",
    },
  });
  eleventyConfig.addPlugin(RenderPlugin);

  return {
    dir: {
      data: "_data",
      includes: "_includes",
      layouts: "_layouts",
    },
  };
}
