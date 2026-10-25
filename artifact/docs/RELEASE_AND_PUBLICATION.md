# Release preparation and publication sequence

This is a prepared release candidate, not an already published GitHub repository or paper. The live private Windows working repository has not been modified by preparing this separate export. No original archive should be deleted merely because this compact package exists.

## Recommended sequence

1. Keep this reviewed public export separate from the private working repository. Its root is the `OperatorMark` folder in the release ZIP; it contains `README.md`, `LICENSE`, `CITATION.cff`, `reproduce.py`, `analysis/`, `docs/`, `research_source/`, `campaign_source/`, `environment/` and `provenance/`.
2. Write a solid preprint using the existing results and limitations. Do not delay the first public research release until journal or conference acceptance. Do not add experiments merely to make the narrative uniformly positive.
3. When the paper is ready, create the public repository from this clean export, add the final repository URL to the citation metadata and paper, and create a versioned release. Do not push the private repository's old Git history or operational archives.
4. Submit the preprint. Once its permanent public identifier is available, add the paper link to the README and citation metadata and record the exact release commit/version in the paper. If a revision changes scientific claims, version the artifact and record it; do not silently alter frozen measurements.
5. Publish the LinkedIn announcement after both the code/results and paper links are live. The submission/upload dates need not be identical. This produces one coherent research announcement while preserving clear attribution.

arXiv performs moderation before announcement and states that processing may take 1–4 days or longer. Coordinate the public announcement around live links rather than promising an exact upload day. [Official arXiv submission availability guidance](https://info.arxiv.org/help/availability.html).

If a workshop or conference is selected, check that venue's current preprint/anonymity requirements before submission. A preprint is not peer review, and this repository must not claim acceptance that has not occurred.

## Items intentionally not invented

The GitHub account/repository URL, final paper title, author list beyond the named project author, ORCID, affiliation, DOI/arXiv identifier, release commit hash and public publication date are not supplied by the evidence. `CITATION.cff` therefore gives a software citation without fabricated identifiers. Add actual values at launch; do not use placeholder identifiers in a public citation.

## License and future commercial work

MIT licensing is prepared for the original included material. It requires preservation of its notice while allowing reuse, modification and commercial use. It is not a mechanism to prevent all copying or guarantee scholarly credit. A dated paper, clear author attribution, versioned artifacts and usable citation metadata provide a concrete scholarly record. This release does not create a paid service or promise one.

## Exact completion boundary

The release preparation task packages the completed adopted numerical evidence and tests the compact statistical replay. It does not publish anything, perform a new campaign, satisfy deferred formal engineering assurance, or make private raw assets public. The next substantive work is writing the paper; public launch follows when the paper and repository are ready.
