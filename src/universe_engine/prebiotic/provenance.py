"""Sourced reference values and explicit assumptions for Milestone C."""
SOURCED = {
    "peptide_bond_dG": {"value_kcal_mol": 3.5, "value_J_mol": 3.5 * 4184.0,
        "source": "American Peptide Society, 'Formation of the Peptide Bond: Condensation and Thermodynamics' (approx +3.5 kcal/mol per peptide bond at physiological pH and temperature); fetched 2026-10-31, secondary source",
        "url": "https://www.americanpeptidesociety.org/knowledge-base/article/21-formation-of-the-peptide-bond-condensation-and-thermodynamics/"},
    "raf_theory": {"source": "Hordijk & Steel; RAF (reflexively autocatalytic and F-generated) sets and the polynomial-time RAF algorithm (search results confirmed the literature exists; the algorithm here follows the standard definition, implemented from understanding, not from a fetched text)"},
}
ASSUMED = {
    "peptide_dS": "reaction entropy set to 0 so dG is temperature independent (assumption; real dS is non-zero)",
    "hydrolysis_rate": "uncatalysed amide hydrolysis rate constants default to a placeholder; real values are extremely slow (half-lives of years or longer, recalled and NOT verified here)",
    "catalytic_boost": "multiplier standing in for unspecified catalysis/activation chemistry; an INPUT, not a prediction",
    "cycling": "wet/dry cycling is an imposed environmental schedule with assumed concentration factor, water activity and temperatures",
    "random_catalysis": "in the random polymer model, which molecule catalyses which reaction is assigned at random with probability p (Kauffman / Hordijk-Steel style); the emergence of RAFs is a property of that assumption",
    "vesicle_geometry": "division at area/sphere-area = 2^(1/3) is geometric (two equal spheres of conserved total area and volume); lysis strain ~4% is a recalled order of magnitude, UNVERIFIED",
    "vesicle_uptake": "amphiphile uptake law and the critical vesicle concentration are placeholders (the primary source could not be fetched); not measured values",
}
