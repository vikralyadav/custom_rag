import * as cheerio from "cheerio";
import fetch from "node-fetch";
import { Document } from "@langchain/core/documents";


import {config} from "dotenv";
config();




import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";



import { createRetrieverTool } from "@langchain/classic/tools/retriever";



import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";

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
// console.log(docs[0]);



const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const docSplits = await textSplitter.splitDocuments(docs);



console.log(`Split into ${docSplits.length} documents`);
// console.log(docSplits[0]);



const api_key = process.env.OPENAI_API_KEY;


const vectorStore = await MemoryVectorStore.fromDocuments(
  docSplits,
  new OllamaEmbeddings({model: "all-minilm:latest"}),
);

const retriever = vectorStore.asRetriever();



console.log("Setup complete. You can now use the retriever to fetch relevant documents.");




const tool = createRetrieverTool(
  retriever,
  {
    name: "retrieve_blog_posts",
    description:
      "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
  },
);
const tools = [tool];


console.log("Created retriever tool:", tools[0].name);

