import torch
from typing import Annotated, TypedDict, Literal
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, trim_messages, AIMessage, HumanMessage, ToolCall

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import chain
from uuid import uuid4
import re
import matplotlib.pyplot as plt
import PIL.Image as Image
import gradio as gr
import spaces   

from rdkit import Chem
from rdkit.Chem import AllChem, QED
from rdkit.Chem import Draw
from rdkit import rdBase
from rdkit.Chem import rdMolAlign
import os
from rdkit import RDConfig
from rdkit.Chem.Features.ShowFeats import _featColors as featColors
from rdkit.Chem.FeatMaps import FeatMaps
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import base64

eleven_key = os.getenv("eleven_key")

elevenlabs = ElevenLabs(api_key=eleven_key)

fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef'))

fmParams = {}
for k in fdef.GetFeatureFamilies():
    fparams = FeatMaps.FeatMapParams()
    fmParams[k] = fparams

device = "cuda" if torch.cuda.is_available() else "cpu"

hf = HuggingFacePipeline.from_model_id(
    #model_id= "swiss-ai/Apertus-8B-Instruct-2509",
    model_id= "microsoft/Phi-4-mini-instruct",
    task="text-generation",
    pipeline_kwargs = {"max_new_tokens": 700, "temperature": 0.1})

chat_model = ChatHuggingFace(llm=hf)

class State(TypedDict):
  '''
    The state of the agent.
  '''
  messages: Annotated[list, add_messages]
  query_smiles: str
  query_task: str
  query_path: str
  query_reference: str
  tool_choice: tuple
  which_tool: int
  props_string: str
  #(Literal["lipinski_tool", "substitution_tool", "pharm_feature_tool"],
  #                   Literal["lipinski_tool", "substitution_tool", "pharm_feature_tool"])


def substitution_node(state: State) -> State:
  '''
    A simple substitution routine that looks for a substituent on a phenyl ring and
    substitutes different fragments in that location. Returns a list of novel molecules and their
    QED score (1 is most drug-like, 0 is least drug-like).

      Args:
        smiles: the input smiles string
      Returns:
        new_smiles_string: a string of novel molecules and their QED scores.
  '''
  print("substitution tool")
  print('===================================================')

  smiles = state["query_smiles"]
  current_props_string = state["props_string"]

  new_fragments = ["c(Cl)c", "c(F)c", "c(O)c", "c(C)c", "c(OC)c", "c([NH3+])c",
                   "c(Br)c", "c(C(F)(F)(F))c"]

  new_smiles = []
  for fragment in new_fragments:
    m = re.findall(r"c(\D\D*)c", smiles)
    if len(m) != 0:
      for group in m:
        #print(group)
        if fragment not in group:
          new_smile = smiles.replace(group[1:], fragment)
          new_smiles.append(new_smile)

  qeds = []
  for new_smile in new_smiles:
    qeds.append(get_qed(new_smile))
  original_qed = get_qed(smiles)

  new_smiles_string = "Substitution or Analogue creation tool results: \n"
  new_smiles_string += f"The original molecule SMILES was {smiles} with QED {original_qed}.\n"
  new_smiles_string += "Novel Molecules or Analogues and QED values: \n"
  for i in range(len(new_smiles)):
    new_smiles_string += f"SMILES: {new_smiles[i]}, QED: {qeds[i]:.3f}\n"
  new_mols = [Chem.MolFromSmiles(x) for x in new_smiles]
  if len(new_smiles) > 0:
    img = Draw.MolsToGridImage(new_mols, molsPerRow=3, subImgSize=(200,200), legends=[f"QED: {qeds[i]:.3f}" for i in range(len(new_smiles))])
    img.save('Substitution_image.png')
  else:
    new_smiles_string += "No valid substitutions were found.\n"

  print(new_smiles_string)
  current_props_string += new_smiles_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def get_qed(smiles):
  '''
    Helper function to compute QED for a given molecule.
      Args:
        smiles: the input smiles string
      Returns:
        qed: the QED score of the molecule.
  '''
  mol = Chem.MolFromSmiles(smiles)
  qed = Chem.QED.default(mol)

  return qed

def lipinski_node(state: State) -> State:
  '''
    A tool to calculate QED and other lipinski properties of a molecule.
      Args:
        smiles: the input smiles string
      Returns:
        props_string: a string of the QED and other lipinski properties of the molecule,
                      including Molecular Weight, LogP, HBA, HBD, Polar Surface Area,
                      Rotatable Bonds, Aromatic Rings and Undesireable Moieties.
  '''
  print("lipinski tool")
  print('===================================================')

  smiles = state["query_smiles"]
  current_props_string = state["props_string"]

  mol = Chem.MolFromSmiles(smiles)
  qed = Chem.QED.default(mol)

  p = Chem.QED.properties(mol)
  mw = p[0]
  logP = p[1]
  hba = p[2]
  hbd = p[3]
  psa = p[4]
  rb = p[5]
  ar = p[6]
  um = p[7]

  props_string = "Lipinski tool results: \n"
  props_string += f'''QED and other lipinski properties of the molecule:
    SMILES: {smiles},
    QED: {qed:.3f},
    Molecular Weight: {mw:.3f},
    LogP: {logP:.3f},
    Hydrogen bond acceptors: {hba},
    Hydrogen bond donors: {hbd},
    Polar Surface Area: {psa:.3f},
    Rotatable Bonds: {rb},
    Aromatic Rings: {ar},
    Undesireable moieties: {um}
  '''

  current_props_string += props_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def pharmfeature_node(state: State) -> State:
  '''
    A tool to compare the pharmacophore features of a query molecule against
    a those of a reference molecule and report the pharmacophore features of both and the feature
    score of the query molecule.

      Args:
        known_smiles: the reference smiles string
        test_smiles: the query smiles string
      Returns:
        props_string: a string of the pharmacophore features of both molecules and the feature
                      score of the query molecule.
  '''
  print("pharmfeature tool")
  print('===================================================')

  test_smiles = state["query_smiles"]
  known_smiles = state["query_reference"]
  current_props_string = state["props_string"]

  smiles = [known_smiles, test_smiles]
  mols = [Chem.MolFromSmiles(x) for x in smiles]

  mols = [Chem.AddHs(m) for m in mols]
  ps = AllChem.ETKDGv3()

  for m in mols:
      AllChem.EmbedMolecule(m,ps)

  o3d = rdMolAlign.GetO3A(mols[1],mols[0])
  o3d.Align()

  keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 'ZnBinder', 'Aromatic', 'LumpedHydrophobe')
  feat_hash = {'Donor': 'Hydrogen bond donors', 'Acceptor': 'Hydrogen bond acceptors',
               'NegIonizable': 'Negatively ionizable groups', 'PosIonizable': 'Positively ionizable groups',
               'ZnBinder': 'Zinc Binders', 'Aromatic': 'Aromatic rings', 'LumpedHydrophobe': 'Hydrophobic/non-polar groups' }

  feat_vectors = []
  for m in mols:
      rawFeats = fdef.GetFeaturesForMol(m)
      feat_vectors.append([f for f in rawFeats if f.GetFamily() in keep])

  feat_maps = [FeatMaps.FeatMap(feats = x,weights=[1]*len(x),params=fmParams) for x in feat_vectors]
  test_score = feat_maps[0].ScoreFeats(feat_maps[1].GetFeatures())/(feat_maps[0].GetNumFeatures())

  feats_known = {}
  feats_test = {}
  for feat in feat_vectors[0]:
    if feat.GetFamily() not in feats_known.keys():
      feats_known[feat.GetFamily()]  = 1
    else:
      feats_known[feat.GetFamily()] += 1

  for feat in feat_vectors[1]:
    if feat.GetFamily() not in feats_test.keys():
      feats_test[feat.GetFamily()]  = 1
    else:
      feats_test[feat.GetFamily()] += 1

  props_string = "PharmFeature tool results: \n"
  props_string += f"The Pharmacophore Feature Overlap Score of the test molecule \
versus the reference molecule is {test_score:.3f}. \n\n"

  for feat in feats_known.keys():
    props_string += f"There are {feats_known[feat]} {feat_hash[feat]} in the reference molecule. \n"

  for feat in feats_test.keys():
    props_string += f"There are {feats_test[feat]} {feat_hash[feat]} in the test molecule. \n"

  current_props_string += props_string
  state["props_string"] = current_props_string
  state["which_tool"] += 1
  return state

def first_node(state: State) -> State:
  '''
    The first node of the agent. This node receives the input and asks the LLM
    to determine which is the best tool to use to answer the QUERY TASK.

      Input: the initial prompt from the user. should contain only one of more of the following:

             smiles: the smiles string, task: the query task, path: the path to the file,
             reference: the reference smiles

             the value should be separated from the name by a ':' and each field should
             be separated from the previous one by a ','.

             All of these values are saved to the state

      Output: the tool choice
  '''
  query_smiles = None
  state["query_smiles"] = query_smiles
  query_task = None
  state["query_task"] = query_task
  query_path = None
  state["query_path"] = query_path
  query_reference = None
  state["query_reference"] = query_reference
  props_string = ""
  state["props_string"] = props_string

  raw_input = state["messages"][-1].content
  parts = raw_input.split(',')
  for part in parts:
    if 'smiles' in part:
      query_smiles = part.split(':')[1]
      if query_smiles.lower() == 'none':
        query_smiles = None
      state["query_smiles"] = query_smiles
    if 'task' in part:
      query_task = part.split(':')[1]
      state["query_task"] = query_task
    if 'path' in part:
      query_path = part.split(':')[1]
      if query_path.lower() == 'none':
        query_path = None
      state["query_path"] = query_path
    if 'reference' in part:
      query_reference = part.split(':')[1]
      if query_reference.lower() == 'none':
        query_reference = None
      state["query_reference"] = query_reference

  prompt = f'You are given a QUERY_TASK given below and a set of available tools. \
  Your job is to determine which tool(S) if any can accomplish the QUERY_TASK.\n\n \
  Choose only from the tool names listed below.\n \
  If exactly one tool can preform the task, reply with the tool name followed by "#".\n \
  If two toold are required together, reply with both tool names separated by a comma, \
  in a single line followed by a "#".\n \
  If none of the tools can perform the task, reply with "None #".\n \
  Reply with ONLY the tool names followed by "#". Tools:\n \
  lipinski_tool: Calculates the following moelcular properties: Quantitative \
  Estimate of Drug-likeness (QED), Molecular weight, LogP (measures lipophilicity, higher is more lipophilic), \
  HBA, HBD, Polar Surface Area, number of rotatable bonds, number of aromatic rings and Undesireable Moieties. \n \
  substitution_tool: Generates structural analogues of the molecule by substituting \
  different chemical groups on the original molecule. Outputs novel molecules and their \
  QED score (1 is most drug-like, 0 is least drug-like). \n \
  pharm_feature_tool: this tool compares the pharmacophore features of a query molecule against \
  a reference molecule. Rreporting the shared pharmacophore features and similarity feature score. \
  Does not report features unique to either moelcule.' 

  res = chat_model.invoke(prompt)

  tool_choices = str(res).split('<|assistant|>')[1].split('#')[0].strip()
  tool_choices = tool_choices.split(',')
  if len(tool_choices) == 1:
    if tool_choices[0].strip().lower() == 'none':
      tool_choice = (None, None)
    else:
      tool_choice = (tool_choices[0].strip().lower(), None)
  elif len(tool_choices) == 2:
    if tool_choices[0].strip().lower() == 'none':
      tool_choice = (None, tool_choices[1].strip().lower())
    elif tool_choices[1].strip().lower() == 'none':
      tool_choice = (tool_choices[0].strip().lower(), None)
    else:
      tool_choice = (tool_choices[0].strip().lower(), tool_choices[1].strip().lower())
  else:
    tool_choice = (None, None)
  state["tool_choice"] = tool_choice
  state["which_tool"] = 0
  print(f"The chosen tools are: {tool_choice}")

  return state

def loop_node(state: State) -> State:
  '''
    This node accepts the tool returns and decides if it needs to call another
    tool or go on to the parser node.

      Input: the tool returns.
      Output: the next node to call.
  '''
  return state

def parser_node(state: State) -> State:
  '''
    This is the third node in the agent. It receives the output from the tool,
    puts it into a prompt as CONTEXT, and asks the LLM to answer the original
    query.

      Input: the output from the tool.
      Output: the answer to the original query.
  '''
  props_string = state["props_string"]
  query_task = state["query_task"]

  prompt = f'Using only the information provided in the CONTEXT below, \
  answer the QUERY_TASK.\n \
  Your answer must:\n Directly address the QUERY_TASK.\n \
  Use only facts found in the CONTEXT (do not invent information).\n \
  Be concise, precise and logically consistent.\n End your answer with a "#" \
  QUERY_TASK: {query_task}.\n \
  CONTEXT: {props_string}.\n '

  res = chat_model.invoke(prompt)
  return {"messages": res}

def reflect_node(state: State) -> State:
  '''
    This is the fourth node of the agent. It recieves the LLMs previous answer and
    tries to improve it.

      Input: the LLMs last answer.
      Output: the improved answer.
  '''
  previous_answer = state["messages"][-1].content
  props_string = state["props_string"]

  prompt = f'You will revise the PREVIOUS ANSWER below using the tools results \
  which you provided below  \
  INSTRUCTIONS:\n \
  Retain all correct information from the PREVIOUS ANSWER. \
  Incorporate only relevent information from the TOOL RESULTS. \
  Add clarifying or enriching details. \
  Do NOT invent or assume any information that is not present in the input. \
  Improve clarity, precision, factual accuracy, and organisation. \
  Provide a well-structured improved asnwer. \
  End \
  your new answer with a "#" \
  PREVIOUS ANSWER: {previous_answer}.\n \
  TOOL RESULTS: {props_string}. '

  res = chat_model.invoke(prompt)
  return {"messages": res}

def get_chemtool(state):
  '''
  '''
  which_tool = state["which_tool"]
  tool_choice = state["tool_choice"]
  
  if tool_choice is None or tool_choice == (None, None):
    return None
  
  if which_tool == 0 or which_tool == 1:
    current_tool = tool_choice[which_tool]
    if current_tool is None:
      return None
  elif which_tool > 1:
    current_tool = None

  return current_tool

def pretty_print(answer):
  final = str(answer['messages'][-1]).split('<|assistant|>')[-1].split('#')[0].strip("n").strip('\\').strip('n').strip('\\')
  for i in range(0,len(final),100):
    print(final[i:i+100])

def print_short(answer):
  for i in range(0,len(answer),100):
    print(answer[i:i+100])

builder = StateGraph(State)
builder.add_node("first_node", first_node)
builder.add_node("substitution_node", substitution_node)
builder.add_node("lipinski_node", lipinski_node)
builder.add_node("pharmfeature_node", pharmfeature_node)
builder.add_node("loop_node", loop_node)
builder.add_node("parser_node", parser_node)
builder.add_node("reflect_node", reflect_node)

builder.add_edge(START, "first_node")
builder.add_conditional_edges("first_node", get_chemtool, {
    "substitution_tool": "substitution_node",
    "lipinski_tool": "lipinski_node",
    "pharm_feature_tool": "pharmfeature_node",
    None: "parser_node"})

builder.add_edge("lipinski_node", "loop_node")
builder.add_edge("substitution_node", "loop_node")
builder.add_edge("pharmfeature_node", "loop_node")

builder.add_conditional_edges("loop_node", get_chemtool, {
    "substitution_tool": "substitution_node",
    "lipinski_tool": "lipinski_node",
    "pharm_feature_tool": "pharmfeature_node",
    None: "parser_node"})

builder.add_edge("parser_node", "reflect_node")
builder.add_edge("reflect_node", END)

graph = builder.compile()

@spaces.GPU
def PropAgent(task, smiles, reference):

  #if Substitution_image.png exists, remove it
  if os.path.exists('Substitution_image.png'):
    os.remove('Substitution_image.png')

  input = {
    "messages": [
        HumanMessage(f'query_smiles: {smiles}, query_task: {task}, query_reference: {reference}')
    ]
  }
  #print(input)

  replies = []
  for c in graph.stream(input): #, stream_mode='updates'):
    m = re.findall(r'[a-z]+\_node', str(c))
    if len(m) != 0:
      reply = c[str(m[0])]['messages']
      if 'assistant' in str(reply):
        reply = str(reply).split("<|assistant|>")[-1].split('#')[0].strip()
        replies.append(reply)
  #check if image exists
  if os.path.exists('Substitution_image.png'):
    img_loc = 'Substitution_image.png'
    img = Image.open(img_loc)
  #else create a dummy blank image
  else:
    img = Image.new('RGB', (250, 250), color = (255, 255, 255))

  elita_text = replies[-1]

  voice_settings = {
          "stability": 0.37,
          "similarity_boost": 0.90,
          "style": 0.0,
          "speed": 1.05
      }

  audio_stream = elevenlabs.text_to_speech.convert(
      text = elita_text,
      voice_id = 'G5KS88IIzHIX1ogRxdrA',
      model_id = 'eleven_multilingual_v2',
      output_format='mp3_44100_128',
      voice_settings=voice_settings
  )

  audio_converted = b"".join(audio_stream)
  audio = base64.b64encode(audio_converted).decode("utf-8")
  audio_player = f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'

  return replies[-1], img, audio_player

with gr.Blocks(fill_height=True) as forest:
  gr.Markdown('''
              # Properties Agent 
              - uses RDKit to calculate lipinski properties
              - finds pharmacophore similarity between two molecules
              - generated analogues of a molecule
              ''')

  name, smiles = None, None
  with gr.Row():
    with gr.Column():
      smiles = gr.Textbox(label="Molecule SMILES of interest (optional): ", placeholder='none')
      ref = gr.Textbox(label="Reference molecule SMILES of interest (optional): ", placeholder='none')
      task = gr.Textbox(label="Task for Agent: ")
      calc_btn = gr.Button(value = "Submit to Agent")
    with gr.Column():
      props = gr.Textbox(label="Agent results: ", lines=20 )
      pic = gr.Image(label="Molecule")
    voice = gr.HTML()


    calc_btn.click(PropAgent, inputs = [task, smiles, ref], outputs = [props, pic, voice])
    task.submit(PropAgent, inputs = [task, smiles, ref], outputs = [props, pic, voice])

forest.launch(debug=False, mcp_server=True)