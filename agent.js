import * as cheerio from "cheerio";
import fetch from "node-fetch";
import { Document } from "@langchain/core/documents";
import { config } from "dotenv";
config();

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { createRetrieverTool } from "@langchain/classic/tools/retriever";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";

import * as z from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { AIMessage, ToolMessage, HumanMessage } from "@langchain/core/messages";
import { StateGraph, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { Ollama } from "@langchain/community/llms/ollama";

// --------------------------
// Custom Web Loader
// --------------------------
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

// --------------------------
// Load and split documents
// --------------------------
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

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});
const docSplits = await textSplitter.splitDocuments(docs);
console.log(`Split into ${docSplits.length} documents`);


const vectorStore = await MemoryVectorStore.fromDocuments(
  docSplits,
  new OllamaEmbeddings({ model: "all-minilm:latest" })
);
const retriever = vectorStore.asRetriever();

const tool = createRetrieverTool(retriever, {
  name: "retrieve_blog_posts",
  description:
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
});
const tools = [tool];

console.log("Retriever tool created:", tools[0].name);


async function generateQueryOrRespond(state) {
  const { messages } = state;
  const model = new Ollama({
    baseUrl: "http://localhost:11434",
    model: "mistral",
    temperature: 0,
  });

  const response = await model.invoke(messages);
  return { messages: [response] };
}

const gradePrompt = ChatPromptTemplate.fromTemplate(
  `You are a grader assessing relevance of retrieved docs to a user question.
  Here are the retrieved docs:
  \n ------- \n
  {context}
  \n ------- \n
  Here is the user question: {question}
  If the docs are relevant to the question, say "yes". Otherwise, say "no".`
);

async function gradeDocuments(state) {
  const { messages } = state;
  const model = new Ollama({
    baseUrl: "http://localhost:11434",
    model: "mistral",
    temperature: 0,
  });

  const result = await gradePrompt.pipe(model).invoke({
    question: messages.at(0)?.content,
    context: messages.at(-1)?.content,
  });

  const content = result.content.trim().toLowerCase();
  return content.includes("yes") ? "generate" : "rewrite";
}

const rewritePrompt = ChatPromptTemplate.fromTemplate(
  `Look at the input and try to reason about the underlying semantic intent / meaning.
  Here is the initial question:
  \n ------- \n
  {question}
  \n ------- \n
  Formulate an improved question:`
);

async function rewrite(state) {
  const { messages } = state;
  const question = messages.at(0)?.content;
  const model = new Ollama({
    baseUrl: "http://localhost:11434",
    model: "mistral",
    temperature: 0,
  });

  const response = await rewritePrompt.pipe(model).invoke({ question });
  return { messages: [response] };
}

async function generate(state) {
  const { messages } = state;
  const question = messages.at(0)?.content;
  const context = messages.at(-1)?.content;

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know, say you don't know.
    Answer in three concise sentences maximum.
    Question: {question}
    Context: {context}`
  );

  const llm = new Ollama({
    baseUrl: "http://localhost:11434",
    model: "mistral",
    temperature: 0,
  });

  const ragChain = prompt.pipe(llm);
  const response = await ragChain.invoke({ context, question });
  return { messages: [response] };
}

function shouldRetrieve(state) {
  const { messages } = state;
  const last = messages.at(-1);
  if (AIMessage.isInstance(last) && last.tool_calls?.length) {
    return "retrieve";
  }
  return END;
}


const GraphState = z.object({
  messages: z.array(z.any()).default([]),
  input: z.string().optional(),
  output: z.string().optional(),
  context: z.any().optional(),
});

const toolNode = new ToolNode(tools);

const builder = new StateGraph(GraphState)
  .addNode("generateQueryOrRespond", generateQueryOrRespond)
  .addNode("retrieve", toolNode)
  .addNode("gradeDocuments", gradeDocuments)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate)
  .addEdge(START, "generateQueryOrRespond")
  .addConditionalEdges("generateQueryOrRespond", shouldRetrieve)
  .addEdge("retrieve", "gradeDocuments")
  .addConditionalEdges("gradeDocuments", (_, result) => result)
  .addEdge("generate", END)
  .addEdge("rewrite", "generateQueryOrRespond");

const graph = builder.compile();


const inputs = {
  messages: [new HumanMessage("What does Lilian Weng say about types of reward hacking?")],
};

for await (const output of await graph.stream(inputs)) {
  for (const [key, value] of Object.entries(output)) {
    const lastMsg = value.messages[value.messages.length - 1];
    console.log(`Output from node: '${key}'`);
    console.log({
      type: lastMsg.type, 
      content: lastMsg.content,
      tool_calls: lastMsg.tool_calls,
    });
    console.log("---\n");
  }
}

