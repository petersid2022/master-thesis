# Βελτιστοποίηση Παραγωγής Κειμένου από Μεγάλα Γλωσσικά Μοντέλα (LLM) μέσω της Τεχνικής Speculative Sampling.
> Optimizing Text Generation with Large language Models via Speculative Sampling.

> Η εργασία αυτή μελετά την τεχνική speculative sampling, με σκοπό την επιλογή ενός μικρού γλωσσικού μοντέλου (SLM) αντί ενός μεγάλου (LLM), όταν το SLM μπορεί να αποδώσει εξίσου καλά, ενεργοποιώντας το LLM μόνο όταν κρίνεται απαραίτητο. Θα διεξαχθούν πειράματα για τη σύγκριση διαφορετικών προσεγγίσεων, με μετρήσεις χρόνου απόκρισης και κατανάλωσης ενέργειας σε σχέση με τα αρχικά μεγάλα μοντέλα. Ιδιαίτερη έμφαση θα δοθεί (α) στην επιλογή των κατάλληλων μοντέλων και (β) στην επίδρασή τους στην ακρίβεια και στην αποδοτικότητα των αποτελεσμάτων.

## Keywords: 
> We treat speculative sampling as an inference scheduling problem, drawing from compiler optimization theory (e.g., branch prediction, PGO, instruction scheduling) to minimize latency and energy usage.
> A compiler-inspired runtime for speculative text generation with multi-model pipelines
> Architect it as a prediction-and-scheduling runtime with compiler-like heuristics

* CPU scheduler
* Compiler-inspired runtime design
* Code generation for inference graphs
* SLM–LLM pipeline as an IR (intermediate representation)

You final thesis produces:
> An adaptive runtime scheduler (like a JIT) for inference.
> A profiling framework that drives decisions.
> Performance/energy trade-off curves similar to compiler optimization trade-offs.

## LINKS
1. Accelerating Large Language Model Decoding with Speculative Sampling (DeepMind)
https://arxiv.org/abs/2302.01318

2. Looking Back at Speculative Decoding (link)
https://news.ycombinator.com/item?id=43216518

3. Instantaneous Grammatical Error Correction with Shallow Aggressive Decoding
https://arxiv.org/abs/2106.04970

4. A Hitchhiker’s Guide to Speculative Decoding - By Team PyTorch at IBM
https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/

5. Speculative Decoding for 2x Faster Whisper Inference
https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/speculative_decoding.ipynb#scrollTo=baf87589-b7fe-45dd-a6f6-9b9223581562

6. Learning Harmonized Representations for Speculative Sampling
https://arxiv.org/abs/2408.15766

// vim: wrap
