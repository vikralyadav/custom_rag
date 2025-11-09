import * as cheerio from "cheerio";
import fetch from "node-fetch";
import { Document } from "@langchain/core/documents";


import {config} from "dotenv";
config();




import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";



import { createRetrieverTool } from "@langchain/classic/tools/retriever";



import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";






import * as z from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { AIMessage } from "@langchain/core/messages";


import { ToolMessage } from "@langchain/core/messages";



import { StateGraph, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";




import { HumanMessage } from "@langchain/core/messages";
import { Ollama } from "@langchain/community/llms/ollama";

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





const API_KEY=process.env.GOOGLE_API_KEY;




// const model = new ChatOpenAI({
//     model: "gpt-4o",
//     temperature: 0,
//   }).bindTools(tools);  



async function generateQueryOrRespond(state) {
  const { messages } = state;


const model = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "mistral", 
  temperature: 0,
});
  

  const response = await model.invoke(messages);
  return {
    messages: [response],
  };
}


console.log("Initialized model with tools.");




// const input = { messages: [new HumanMessage("hello!")] };
// const result = await generateQueryOrRespond(input);
// console.log(result.messages[0]);



// const input = {
//   messages: [
//     new HumanMessage("What does Lilian Weng say about types of reward hacking?")
//   ]
// };
// const result = await generateQueryOrRespond(input);
// console.log(result.messages[0]);




const prompt = ChatPromptTemplate.fromTemplate(
  `You are a grader assessing relevance of retrieved docs to a user question.
  Here are the retrieved docs:
  \n ------- \n
  {context}
  \n ------- \n
  Here is the user question: {question}
  If the content of the docs are relevant to the users question, score them as relevant.
  Give a binary score 'yes' or 'no' score to indicate whether the docs are relevant to the question.
  Yes: The docs are relevant to the question.
  No: The docs are not relevant to the question.`,
);

const gradeDocumentsSchema = z.object({
  binaryScore: z.string().describe("Relevance score 'yes' or 'no'"),  
})

async function gradeDocuments(state) {
  const { messages } = state;

 const model = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "mistral", 
  temperature: 0,
});

  const score = await prompt.pipe(model).invoke({
    question: messages.at(0)?.content,
    context: messages.at(-1)?.content,
  });

  if (score.binaryScore === "yes") {
    return "generate";
  }
  return "rewrite";
}



// const input = {
//   messages: [
//       new HumanMessage("What does Lilian Weng say about types of reward hacking?"),
//       new AIMessage({
//           tool_calls: [
//               {
//                   type: "tool_call",
//                   name: "retrieve_blog_posts",
//                   args: { query: "types of reward hacking" },
//                   id: "1",
//               }
//           ]
//       }),
//       new ToolMessage({
//           content: "meow",
//           tool_call_id: "1",
//       })
//   ]
// }
// const res = await gradeDocuments(input);


// console.log("Grading result:", res);

// const input = {
//   messages: [
//       new HumanMessage("What does Lilian Weng say about types of reward hacking?"),
//       new AIMessage({
//           tool_calls: [
//               {
//                   type: "tool_call",
//                   name: "retrieve_blog_posts",
//                   args: { query: "types of reward hacking" },
//                   id: "1",
//               }
//           ]
//       }),
//       new ToolMessage({
//           content: "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
//           tool_call_id: "1",
//       })
//   ]
// }
// const result = await gradeDocuments(input);


console.log("Grading result:");




const rewritePrompt = ChatPromptTemplate.fromTemplate(
  `Look at the input and try to reason about the underlying semantic intent / meaning. \n
  Here is the initial question:
  \n ------- \n
  {question}
  \n ------- \n
  Formulate an improved question:`,
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
  return {
    messages: [response],
  };
}





// const input = {
//   messages: [
//     new HumanMessage("What does Lilian Weng say about types of reward hacking?"),
//     new AIMessage({
//       content: "",
//       tool_calls: [
//         {
//           id: "1",
//           name: "retrieve_blog_posts",
//           args: { query: "types of reward hacking" },
//           type: "tool_call"
//         }
//       ]
//     }),
//     new ToolMessage({ content: "meow", tool_call_id: "1" })
//   ]
// };

// const response = await rewrite(input);
// console.log("Rewritten question:");
// console.log(response.messages[0].content);




async function generate(state) {
  const { messages } = state;
  const question = messages.at(0)?.content;
  const context = messages.at(-1)?.content;

  const prompt = ChatPromptTemplate.fromTemplate(
  `You are an assistant for question-answering tasks.
      Use the following pieces of retrieved context to answer the question.
      If you don't know the answer, just say that you don't know.
      Use three sentences maximum and keep the answer concise.
      Question: {question}
      Context: {context}`
  );

  const llm = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "mistral", 
  temperature: 0,
});

  const ragChain = prompt.pipe(llm);

  const response = await ragChain.invoke({
    context,
    question,
  });

  return {
    messages: [response],
  };
}



const input = {
  messages: [
    new HumanMessage("What does Lilian Weng say about types of reward hacking?"),
    new AIMessage({
      content: "",
      tool_calls: [
        {
          id: "1",
          name: "retrieve_blog_posts",
          args: { query: "types of reward hacking" },
          type: "tool_call"
        }
      ]
    }),
    new ToolMessage({
      content: "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
      tool_call_id: "1"
    })
  ]
};

const response = await generate(input);
console.log(response.messages[0].content);


const toolNode = new ToolNode(tools);




function shouldRetrieve(state) {
  const { messages } = state;
  const lastMessage = messages.at(-1);

  if (AIMessage.isInstance(lastMessage) && lastMessage.tool_calls.length) {
    return "retrieve";
  }
  return END;
}



// const GraphState = {
//   context: {},
//   input: "",
//   output: "",
// };



const builder = new StateGraph(GraphState)
  .addNode("generateQueryOrRespond", generateQueryOrRespond)
  .addNode("retrieve", toolNode)
  .addNode("gradeDocuments", gradeDocuments)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate)

  .addEdge(START, "generateQueryOrRespond")

  .addConditionalEdges("generateQueryOrRespond", shouldRetrieve)
  .addEdge("retrieve", "gradeDocuments")

  .addConditionalEdges(
    "gradeDocuments",
  
    (state) => {
    
      const lastMessage = state.messages.at(-1);
      return lastMessage.content === "generate" ? "generate" : "rewrite";
    }
  )
  .addEdge("generate", END)
  .addEdge("rewrite", "generateQueryOrRespond");

const graph = builder.compile();





const inputs = {
  messages: [
    new HumanMessage("What does Lilian Weng say about types of reward hacking?")
  ]
};

for await (const output of await graph.stream(inputs)) {
  for (const [key, value] of Object.entries(output)) {
    const lastMsg = output[key].messages[output[key].messages.length - 1];
    console.log(`Output from node: '${key}'`);
    console.log({
      type: lastMsg._getType(),
      content: lastMsg.content,
      tool_calls: lastMsg.tool_calls,
    });
    console.log("---\n");
  }
}