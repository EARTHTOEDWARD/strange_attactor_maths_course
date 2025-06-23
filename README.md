## What is this repository?

**Strange-Attractor-Maths-Course** is a self-contained, code-first roadmap that takes you
from “I barely remember algebra” all the way to **simulating and visualising chaotic
systems** such as the Lorenz and Rössler attractors.

The project marries three elements:

| Layer | Purpose |
|-------|---------|
| **Essential 12 micro-curriculum** | Twelve bite-sized core concepts (PEMDAS → eigenvectors) packaged as Anki cards, cheat-sheet, drills, and “micro-projects.” |
| **Reusable Python library (`src/`)** | Clean, testable implementations of classic maps (logistic, Henon, Lorenz, etc.) plus helper plotters—ready for notebooks or Streamlit dashboards. |
| **Jupyter notebooks (`notebooks/`)** | One notebook per milestone; each begins with a 5-minute theory refresher, then jumps into code so you *see* dynamics immediately. |

Why another chaos repo? Because most chaos texts assume a year of university maths;
this one inverts the funnel—**learn only the atoms you need, exactly when you need
them, and visualise every step.**

### Who is it for?

* Curious coders with rusty maths
* Researchers prototyping dynamical-systems demos
* Educators who want a scaffolded, interactive syllabus
* LLM agents that need a well-documented corpus of clean chaos code

### At-a-Glance Features

* 📚 **Anki deck** (48 cards) auto-generated from the Essential 12  
* 📜 **One-page cheat-sheet** for wall-pin reference  
* 🔁 **Drill generators** (links) producing infinite fresh practice  
* 🖼 **Matplotlib visual hooks**: every concept tied to a picture  
* 🧩 **Micro-projects** under 15 lines of code each  
* 🛠 **Extensible API**: add new maps with two functions (`f`, `Jacobian`) and they
  auto-register in visualisers

### How LLMs can use this repo

* In-context “library” for answering questions on nonlinear dynamics
* Code examples ready for chain-of-thought tool-calling (minimal deps)
* Structured file names & docstrings make AST parsing trivial

> **TL;DR** — Clone, `pip install -r requirements.txt`, run
> `python -m src.visualise --demo`, and watch chaos emerge.

---

Happy hacking & strange-attractor hunting!

