# Shared corpus: clinical knowledge articles

This folder is the one corpus shared by the course modules from week 6 onward.
Modules read it from here; they never copy it. It is the knowledge base of a
medical ERP: the articles a clinical module would hold so that staff can look up a
condition, its symptoms, how it is diagnosed and how it is treated.

The articles are educational summaries written for this practice project. They are
not clinical guidance and must not be used to diagnose or treat anyone.

## Format

Each file is `articles/mNN.txt`:

    Title: Hypertension
    Specialty: Cardiology
    Code: I10
    Updated: 2025

    ## Overview
    one paragraph

    ## Symptoms
    one paragraph

    ## Diagnosis
    one paragraph

    ## Treatment
    one paragraph

- The lines up to the first blank line are metadata: `Title`, `Specialty`, `Code`
  (an ICD-10 code) and `Updated` (year).
- Every `## Heading` starts a section. The four sections are the same in every article,
  so a later week can search one field (for example `symptoms:`) and get a real answer.
- Spelling is British, matching the rest of the course material.

## Coverage

30 articles across 10 specialties. Some topics overlap on purpose. Hypertension is
mentioned in the kidney, stroke, heart-failure and diabetes articles; fever appears
across the infections; inhalers appear in both asthma and COPD; insulin appears in both
diabetes articles. That overlap is what makes ranked and fuzzy results differ from a
simple yes-or-no match.
