<p align="center">
  <img src="assets/logo.png" alt="AgentGL Logo" width="600" />
</p>

<p align="center">
  <img src="assets/framework.png" alt="AgentGL Framework" width="900" />
</p>

---

## To Reviewers

1. **Typos in the paper (Subsection 4.3, Table 2, Table 4).**
   We noticed a few typos in the paper:

   * <span style="color:red"><b>"Training Proces" → "Training Process"</b></span> (Subsection 4.3)
   * <span style="color:red"><b>"SSO" → "MSO"</b></span> (Table 2)
   * <span style="color:red"><b>λₛ → λᵣ</b></span>

   Due to the tight submission deadline, these slipped through during the final revision. We sincerely apologize for the confusion and hope this does not affect your understanding of our method. We will correct them in the final version.

2. **Code usage during the review period.**  
   This codebase is intended only for the review process. We retain the core implementation and replace paths/configs with placeholders for anonymization and sanitization.  
   We will release a one-click runnable version after the review process concludes.

---

## Layout

- `OpenRLHF-RAG/`: OpenRLHF core with graph-native search extensions  
- `train/`: dataset builders and utility scripts  
- `evaluation/`: evaluation drivers and metrics  
- `scripts/`: training and orchestration scripts  

---

## Quick Start (high level)

1. Prepare graph artifacts and model checkpoints (paths are placeholders in scripts).  
2. Run training scripts under `scripts/`.  
3. Use `evaluation/` tools to generate and score results.  

---

## Acknowledgements

1. We thank **[OpenRLHF][openrlhf]** for providing an easy-to-use RL training framework.  
2. We thank **[R1-Searcher][r1-searcher]** for the code implementations of the search-related extensions.  
   Thank you for your wonderful work!

[openrlhf]: https://github.com/OpenRLHF/OpenRLHF
[r1-searcher]: https://github.com/RUCAIBox/R1-Searcher

