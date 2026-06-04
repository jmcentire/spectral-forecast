# Organizational And Leadership Dataset Scout

Date: 2026-06-04

## Selection Rule

For this method, the best organizational/leadership datasets are not ordinary
one-row-per-person survey tables. They should preserve distributed behavior:

- time-stamped communication, contact, contribution, or leadership text;
- multiple people, departments, teams, firms, projects, or roles;
- enough temporal depth to build baselines and residual structure;
- a known organizational layer for post-hoc attribution, such as departments,
  roles, crisis periods, releases, outcomes, or firm events;
- public access and a reproducible download path.

Static HR or attrition tables can be useful downstream as explanatory controls,
but they are weak first-class tests for spectral/stigmergic structure discovery.

## Five-Dataset Expansion Queue

| Priority | Dataset | Organizational lens | Why it fits | First adapter |
| ---: | --- | --- | --- | --- |
| 1 | SNAP `email-Eu-core-temporal` | Internal organizational email | 986 people, 332,334 temporal directed emails over 803 days; department subnetworks support cohort checks, but static department labels do not map onto the temporal full-graph node IDs | Bin email events into global and top-node activity, reciprocity, entropy, and concentration series |
| 2 | SocioPatterns workplace contact data | Face-to-face workplace interaction | High-resolution temporal contacts in an office, plus department metadata; small enough for a quick pilot | Bin 20-second contacts into department activity/contact-intensity time series |
| 3 | Enron email corpus/core | Corporate crisis communication and leadership failure | Public corporate communication corpus around a known collapse; role/hierarchy/crisis period attribution is possible | Start with core temporal sequences or curated weekly graphs; compare pre-crisis/crisis/post-crisis residual structure |
| 4 | MAEC earnings-call dataset | Executive/leadership communication | Speaker-attributed earnings-call transcripts plus low-level audio features; firm/date structure and market-risk context | Convert per-speaker segments into firm-quarter leadership signal series: sentiment, modality, uncertainty, turn-taking, audio prosody proxies |
| 5 | GH Archive / curated open-source org datasets | Distributed leadership and coordination in open-source organizations | Public event streams for issues, PRs, commits, membership, releases; projects have visible maintainers and outcomes | Bin GitHub events by repo/actor/role into coordination-load, response, centralization, and contribution-flow series |

## Candidate Details

### 1. SNAP Email-Eu-Core Temporal

Source:

- https://snap.stanford.edu/data/email-Eu-core-temporal.html

Fit:

- Directed temporal email edges `(source, target, timestamp)`.
- 986 nodes, 332,334 temporal edges, 803-day span.
- Four departmental temporal subnetworks are available.
- Static `email-Eu-core` department labels exist for the static graph, but SNAP
  notes that the static and temporal node IDs are not the same. Do not use the
  static department-label file for full temporal-graph attribution.

Why it is first:

- Small enough for a fast run.
- Directly organizational.
- Already temporal.
- It can be run without labels on the full temporal graph, and the four
  department subnetworks can be used as separate cohort checks.

First question:

> Does the detector surface coherent residual structure around shifts in
> communication load, reciprocity, concentration, or top-node coordination
> bursts?

Suggested first run:

1. Download the full temporal file and, separately, the four temporal
   department subnetworks.
2. Build daily or half-day bins.
3. Emit series for total volume, top-node send/receive volume, sender
   concentration, receiver concentration, reciprocity, and active-node entropy.
4. Run label-free autotune.
5. Attribute top windows to node-level communication structure; compare the
   four department subnetworks as separate cohorts rather than as labels on the
   full graph.

### 2. SocioPatterns Workplace Contact Network

Source:

- https://sociopatterns.org/datasets/test/
- https://sociopatterns.org/datasets

Fit:

- Temporal face-to-face contacts in an office building in France in 2015.
- Contact events are recorded in 20-second intervals.
- Department metadata is provided.
- Data is public-domain dedicated.

Why it is second:

- It is genuinely workplace behavior, not email.
- It is very small, so it should be a quick adapter test.
- Department labels make discovered structure interpretable.

First question:

> Does an agnostic observer recover meeting rhythms, department coupling, or
> cross-department contact regimes from proximity events without being given
> department labels?

Suggested first run:

1. Download contact list and department list.
2. Bin by 5, 10, and 30 minutes.
3. Build series for department contact volume, cross-department contact volume,
   active participants, contact entropy, and bridge contacts.
4. Run nulls that permute department labels and timing separately.

### 3. Enron Email Corpus / Enron Core Temporal Sequences

Sources:

- https://www.cs.cmu.edu/~./enron/
- https://www.wacclearinghouse.org/jwa/corpora/enron/
- https://www.cs.cornell.edu/~arb/data/sos-email-Enron-core/

Fit:

- Large public corporate email corpus.
- Known organizational crisis and collapse context.
- Prior work has used it for social-network and organizational communication
  analysis.
- The sequence-of-sets core version is smaller and more directly suited to
  temporal structure runs.

Why it is third:

- This is the clearest organizational-theory case, but it carries more cleaning
  and ethics overhead than Email-Eu or SocioPatterns.
- The crisis timeline gives an external attribution axis, but it should not be
  treated as a prediction target.

First question:

> Does latent communication structure shift around known organizational crisis
> periods, leadership cliques, or cross-role coordination changes?

Suggested first run:

1. Use the smaller Enron core temporal sequences first.
2. Bin weekly.
3. Build series for sender/recipient activity, recipient-set diversity,
   clique concentration, broker-like cross-group communication, and thread
   burstiness.
4. Attribute top windows against known public crisis timeline after detection.

### 4. MAEC Earnings-Call Dataset

Source:

- https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction

Fit:

- Speaker-attributed earnings-call transcripts and low-level audio features.
- Firm/date structure supports longitudinal leadership communication analysis.
- Executive speech, analyst questions, and Q&A structure give a leadership
  communication surface rather than an internal network surface.

Why it is fourth:

- It is the best direct leadership-communication candidate.
- It needs a text/audio feature adapter before spectral/stigmergic observation.

First question:

> Does latent structure surface changes in executive communication regimes,
> such as uncertainty, evasion, modality, turn-taking, or executive/analyst
> coupling, before any market or risk label is used?

Suggested first run:

1. Start with transcript text and speaker labels, not audio.
2. Group by firm and call date.
3. Build series for executive sentiment/uncertainty/modality, question-answer
   divergence, executive share of speech, CFO/CEO contrast, and lexical
   concentration.
4. Use financial-risk labels only for post-hoc attribution.

### 5. GH Archive / Curated Open-Source Organization Datasets

Sources:

- https://www.gharchive.org/
- https://zenodo.org/records/14230406
- https://zenodo.org/records/18472549

Fit:

- Public GitHub event streams include commits, pull requests, issues, forks,
  membership, releases, and comments.
- The NumFOCUS contributor-activity dataset gives a bounded community of
  scientific open-source projects.
- GitHub organization-network datasets give member/repository contribution
  structure.

Why it is fifth:

- It is organizational/leadership-adjacent through maintainership,
  coordination, review load, release cadence, and contributor centralization.
- It is public and scalable, but attribution requires care because outcomes
  are not as crisp as seizure labels or department metadata.

First question:

> Does the observer surface coordination-regime shifts in open-source projects:
> maintainer bottlenecks, release crunches, contributor churn, issue/PR
> congestion, or governance concentration?

Suggested first run:

1. Start with a bounded curated dataset such as NumFOCUS contributor activity,
   not the full GH Archive firehose.
2. Bin weekly per project.
3. Build series for PR open/close/comment volume, issue latency proxies,
   maintainer concentration, actor entropy, release cadence, and cross-repo
   contributor movement.
4. Attribute top windows to releases, project events, and maintainer-load
   bursts after detection.

## Run Order

Recommended immediate sequence:

1. Email-Eu-core temporal.
2. SocioPatterns workplace.
3. Enron core temporal.
4. NumFOCUS/GitHub organization activity.
5. MAEC leadership communication.

The first two should validate the organizational-network adapter. Enron then
tests known crisis-period structure. GitHub tests distributed coordination and
maintainership. MAEC tests leadership language and call dynamics, which is the
largest adapter jump.

## Method Boundaries

- Do not call these "leadership predictors."
- For communication datasets, the event is not the point. The point is whether
  latent organizational structure surfaces before domain labels are applied.
- Use departments, roles, crisis dates, releases, outcomes, or market labels
  only for attribution or supervised ablations after discovery.
- Report negative runs. If a dataset collapses into simple volume spikes, that
  is a useful boundary.
- Keep static HR/attrition tables as explanatory controls, not core structure
  discovery datasets.
