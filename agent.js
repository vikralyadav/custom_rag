import * as cheerio from "cheerio";
import fetch from "node-fetch";
import { Document } from "@langchain/core/documents";

class CustomCheerioWebLoader {
  constructor(url) {
    this.url = url;
  }

  async load() {
    const res = await fetch(this.url);
    const html = await res.text();
    const $ = cheerio.load(html);

   
    const text = $("body").text().replace(/\s+/g, " ").trim();

    return [new Document({ pageContent: text, metadata: { source: this.url } })];
  }
}

const urls = [
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
];


const docsArrays = await Promise.all(
  urls.map(async (url) => new CustomCheerioWebLoader(url).load())
);


const docs = docsArrays.flat();

console.log(`Loaded ${docs.length} documents`);
console.log(docs[0]);
