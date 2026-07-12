# Deepfakes by the Numbers — A Fact-Checked Report

**Compiled June 2026 · covering data published 2023–2026**
Scope: (1) overall scale, (2) deepfakes in social/"reels" content, (3) AI-generated child sexual abuse material (CSAM), (4) court cases where deepfake video was detected or contested, (5) detection reality check.

---

## 0. How to read these numbers (reliability tiers)

Deepfake statistics circulate widely but vary enormously in quality. Throughout, figures are tagged:

- **[A] Authoritative** — Internet Watch Foundation (IWF), U.S. National Center for Missing & Exploited Children (NCMEC), court records, peer-reviewed studies, government/official testimony.
- **[B] Credible secondary** — major news (Reuters, Bloomberg, NBC, NYT), established analysts (Deloitte, Gartner), academic preprints.
- **[C] Vendor/marketing** — security companies that sell detection or verification tools (DeepStrike, Keepnet, Sumsub, Resemble AI, iProov, Pindrop, Security Hero/Home Security Heroes). Directional and often-cited, but frequently unaudited, sometimes circular (vendors cite each other), and incentivised toward alarming numbers. **Treat [C] as indicative, not precise.**

Two cross-cutting cautions that recur below:
1. **The "8 million deepfakes by 2025" figure is a projection, not a count** — originated by detection firms DeepMedia/DeepStrike, extrapolated from ~500,000 in 2023. [C]
2. **The most-quoted CSAM headline number is badly misleading** — see §3's NCMEC/Amazon caveat. Public awareness of this correction is low.

---

## 1. Overall scale and growth

| Metric | Figure | Source / tier |
|---|---|---|
| Deepfake files online, 2023 → 2025 | ~500,000 → ~8,000,000 (projection) | DeepMedia / DeepStrike [C] |
| Baseline 2019 | ~14,000 videos | Deeptrace/Sensity [C] |
| Annual video growth rate | ~900% | DeepStrike [C] |
| Detected deepfake incidents 2022 → 2023 | ~10× increase | biometric-security firms [C] |
| Q1 2025 reported incidents | 179 (up 19% over all of 2024) | Resemble AI via Variety [B/C] |
| Direct fraud losses, North America Q1 2025 | >$200 million | Resemble AI [C] |
| Direct fraud losses, Q2 2025 | ~$347 million | Resemble AI [C] |
| U.S. gen-AI fraud forecast | $12.3B (2023) → $40B (2027) | Deloitte [B] |
| Largest single deepfake fraud to date | $25.6M (Arup, Hong Kong, Feb 2024, 15 transfers) | Financial Times / CNN [B] |
| Orgs hit by a deepfake in prior 12 months | 62% (n=302) | Gartner, Sep 2025 [B] |

**Human detection is poor.** People correctly identify high-quality deepfake *video* only ~24.5% of the time [C]; an iProov study found just **0.1% of participants** correctly classified every real/fake item shown [C]. A peer-reviewed meta-analysis (56 studies, 86,155 participants, *Computers in Human Behavior Reports*, 2024) put average human accuracy at **55.5%** — barely above chance. **[A]**

---

## 2. Deepfakes in social media / "reels" content

### Where deepfakes are distributed
For Q3 2025 incidents tracked by Resemble AI, the most common host platforms were **YouTube**, then **Instagram (26.8%)**, **Facebook (18.8%)**, **TikTok (18.3%)**, and **WhatsApp (6.3%)**. [C] Short-form video ("reels"/Shorts) is a primary vector for two things:

1. **Scam ads using cloned celebrity likenesses.** Most common lures: fake giveaways, then crypto/investment scams (~30%), weight-loss products (~25%), beauty products (~24%), gadgets (~22%). [C] Some major retailers report **1,000+ AI-generated scam calls per day** spilling over from these campaigns. [B]
2. **Non-consensual intimate imagery (NCII)** — the single largest category of deepfake *content* overall.

### Non-consensual sexual deepfakes (the dominant use)
This is where the numbers are starkest — and where careful sourcing matters:

- **96%** of deepfake videos online were non-consensual sexual content — **this is the 2018–2019 Sensity/Deeptrace figure** (Ajder et al., 2019). **[A, but dated]**
- **98%** of online deepfake videos were NCII — the **2023** "State of Deepfakes" report by Security Hero / Home Security Heroes. **[C, more recent]**
- ~**99–100%** of victims in sexual deepfakes are **women/girls**; on the top sites, ~99% of those targeted are celebrities. **[A]** (Caveat: site-scraping methods under-count private, non-celebrity victims whose abuse spreads via DMs/email rather than public sites.)
- **MrDeepFakes** (now defunct) hosted **43,000 sexual deepfake videos** of **~3,800 individuals**, watched **~1.5 billion times**; 95.3% of those depicted were women entertainers, with many K-pop figures (Han et al., 2025, peer-reviewed). **[A]**
- **"Nudify"/undress apps**: across 85 such sites, ~**18.5 million** combined monthly visitors over a six-month period (Mantzarlis & Lakatos, 2025). **[B]**
- **Population prevalence**: in a 10-country survey of 16,000+ people, **2.2%** reported being victims of deepfake pornography and **1.8%** reported perpetrating it (Umbach et al., CHI 2024, peer-reviewed). **[A]**

### Notable social-content incidents
- **Taylor Swift, January 2024**: AI-generated sexual images spread on X reached an estimated **47 million views** before removal. **[B]**
- **Telegram "nudify" bots** reached roughly **4 million monthly users** by late 2024 (South Korea focus). **[B/C]**
- **South Korea** logged about **297 deepfake sex-crime cases** in seven months of 2024 — nearly double 2021. **[B]**
- **Schools (U.S.)**: cases in New Jersey (the Francesca Mani / Westfield case that prompted state law), Florida, Washington, Kentucky — plus internationally in Spain and South Korea — involved students making sexual deepfakes of classmates. **[A/B]**

---

## 3. AI-generated child sexual abuse material (CSAM)

> This section reports the scale of a crime and the legal response. All figures are from child-protection authorities and courts.

### Internet Watch Foundation (UK) — 2025 data **[A]**
- **2025 was the worst year on record**: the IWF actioned **312,030 reports** confirmed to contain CSAM (+7% over 2024).
- **AI CSAM videos exploded**: about **3,440 AI videos** of child sexual abuse found in 2025, versus **13** in 2024 — roughly a **260-fold** rise (the IWF states the increase as ~26,362%; it cites both 3,440 and 3,443 across releases).
- In total, **8,029 AI-generated images and videos** were assessed as realistic CSAM in 2025.
- **65%** of the AI videos were **Category A** (the most severe UK classification — penetration, sadism, bestiality), versus 43% for non-AI material. AI content is disproportionately extreme.
- **First half of 2025**: 210 webpages of AI CSAM (up from 42 a year earlier, ~**400%**), containing **1,286 videos** (up from just 2). (Bloomberg/IWF.)
- **Technique**: a fine-tuning method (LoRA) can build a realistic deepfake of a **specific real child from as few as ~20 images in ~15 minutes**.
- **Gendered**: ~**97%** of victims in AI CSAM are girls.

### NCMEC (U.S.) CyberTipline — and a critical fact-check **[A]**
- **2023 → 2024**: reports involving generative AI rose from **4,700 to 67,000** (**+1,325%**).
- **2025**: NCMEC received **more than 1.5 million reports** with a generative-AI nexus.

**⚠️ The most important caveat in this entire report:** Of those 1.5 million, **~1.1 million were submitted by Amazon AI Services and contained no actionable information** — they were automated **hash-matches against AI *training* data**, not confirmed AI-generated CSAM. A Stanford Internet Observatory analysis (via a Bloomberg investigation) found that **at least 78%** of the "Generative AI" checkbox reports in H1 2025 **did not involve any AI-generated CSAM at all.** Headlines stating AI CSAM is "flooding the internet" (e.g., a 2024 NYT headline) over-read a single ambiguous checkbox. **[A/B]**

**Excluding the Amazon training-data reports, NCMEC's 2025 actionable breakdown:**
- ~**12,000** reports: CSAM found inside AI training datasets
- ~**7,000**: users generating or possessing AI-generated CSAM
- ~**30,000**: users attempting to generate CSAM via image uploads + text prompts
- ~**145,000**: users altering/manipulating existing CSAM files with AI
- ~**3,000**: other AI-facilitated exploitation (e.g., chat-based grooming)

So the real signal is large and rising — but it is **tens of thousands of actionable cases**, not 1.5 million pieces of AI CSAM.

### Convictions and law **[A]**
- **Hugh Nelson (UK)** — sentenced **October 2024 to 18 years** (plus 6 on extended licence) at Bolton Crown Court after pleading guilty to 16 child-sexual-abuse offences. He used the 3D character tool **Daz 3D** to turn ordinary photos of real children into abuse imagery, selling them (~£5,000 over ~18 months, ~£80/image) in encrypted chatrooms. Regarded as the **UK's first conviction for creating AI/computer-generated CSAM** — establishing that the law treats AI CSAM identically to photographic CSAM.
- **UK** (2025): new Crime and Policing Bill offence for making/possessing/supplying a "CSA image-generator," and moves to ban "nudify" tools.
- **U.S.**: the **TAKE IT DOWN Act** (signed May 2025) criminalises knowingly publishing non-consensual intimate imagery (including AI), with **48-hour platform takedown** duties; per Public Citizen, **~45 states** have enacted intimate-deepfake laws, many minor-focused.

---

## 4. Court cases where deepfake video was detected or contested

Two distinct legal problems are emerging: **(a) fabricated AI evidence submitted as real**, and **(b) the "deepfake defense"** — claiming *genuine* evidence is fake (the "liar's dividend," a term coined by law professors Chesney & Citron in 2018).

### (a) Detected fabricated AI evidence
- **Mendones v. Cushman & Wakefield** (California) — Judge **Victoria Kolakowski** noticed a "witness" video looked wrong (near-motionless face, odd cuts, repeated mannerisms) and identified it as **AI-generated audio/video of a real person** submitted by self-represented plaintiffs as authentic testimony. Reported as **one of the first instances where a deepfake was submitted as genuine evidence and caught**. **[A/B]**
- **Alameda County, California (Sept 9, 2025)** — a judge **threw out a case and sanctioned the plaintiffs** for "intentionally submitting false evidence" produced with AI. **[A/B]**
- **Florida** — a woman spent **two days in jail** after an ex allegedly fabricated AI-generated text messages; charges were dropped only after ~8 months. **[B]**

### (b) The "deepfake defense" — mostly rejected so far
- **Huang v. Tesla** — Tesla declined to admit a real video of Elon Musk discussing Autopilot, citing deepfake risk. Judge **Evette Pennypacker** rebuked the position, warning that famous people can't "hide behind" deepfake claims to disown real statements, and ordered Musk to testify. **[A/B]**
- **Wisconsin v. Rittenhouse (2021)** — defense argued Apple's pinch-to-zoom uses AI to "manipulate" footage; the court put the burden on the prosecution to prove otherwise, and (lacking a ready expert) it couldn't zoom in. An early, influential example of AI-skepticism shaping a trial. **[A]**
- **U.S. v. Reffitt** and other **January 6** prosecutions — defendants suggested footage could be AI-manipulated (one cited a 2017 Obama research deepfake). Claims failed; **defendants were convicted**. **[A/B]**
- **U.S. v. Khalilian** — defense sought to exclude a voice recording as possibly deepfaked; the court indicated a familiar witness identifying the voice was "probably enough" to admit it — illustrating that **traditional authentication rules strain against deepfakes**. **[A]**
- **Valenti v. Dfinity** — a plaintiff's unsupported deepfake allegation about video backfired, with the court siding against them. **[A]**

### The institutional response
A November 2025 report from **20 experts** (covered by CU Boulder) urged: specialised AI training for judges and jurors, national standards for AI-enhanced/AI-generated footage, and better evidentiary-video storage. Proposed **Federal Rule of Evidence 707** would hold machine-generated evidence to reliability standards; other proposals (FRE 901(c), 901(b)(11)) would tighten authentication and shift deepfake-authenticity calls from jury to judge. Courts warn detection is **probabilistic and immature**, and that requiring forensic review of every disputed clip could **widen access-to-justice gaps** for parties who can't afford experts. **[A/B]**

---

## 5. Detection reality check

- **In-the-wild accuracy is much lower than lab claims.** The **DeepFake-Eval-2024** benchmark (real social-media deepfakes) found commercial detectors around **~78%** on in-the-wild content, with many models dropping **45–50%** versus clean datasets. **[B]**
- A **UC San Diego** team (Aug 2025) reported **98.3%** on AI-generated video — but **in controlled evaluations**. **[B]**
- Detectors are **brittle**: high on clean data, collapsing on filtered/adversarial real-world inputs, and needing recalibration each time a new generator ships. **[A/B]**
- Audio detector EERs in benchmarks are often **>13%** — far from solved. **[C]**

**Implication:** the gap between *generation* and *reliable detection* is widening, which is exactly why courts, platforms, and child-protection bodies increasingly favour **provenance/authentication** (signing real content) over **post-hoc detection** (catching fakes).

---

## 6. Bottom line — what's solid vs. shaky

**Solid [A]:**
- AI CSAM is rising fast and is disproportionately extreme: IWF's **~3,440 AI abuse videos in 2025 vs 13 in 2024**, **65% Category A**, **97% girl victims**.
- A **real person has been imprisoned (18 yrs, UK)** specifically for AI-generated CSAM.
- Courts have **already detected** a deepfake submitted as authentic evidence (Mendones) and have **rejected** opportunistic "deepfake defenses" (Tesla, Jan 6).
- Sexual content dominates malicious deepfakes, and victims are overwhelmingly women/girls (multiple peer-reviewed sources).
- Humans cannot reliably detect deepfakes (~55% in meta-analysis; ~24.5% for high-quality video).

**Treat with caution [C] / commonly misreported:**
- **"8 million deepfakes by 2025"** — a vendor *projection*, not a measured count.
- **"1.5 million AI CSAM reports to NCMEC in 2025"** — ~1.1M were Amazon training-data hash matches; **≥78%** of the AI-checkbox reports didn't involve AI-generated CSAM. The real actionable figure is in the **tens of thousands**.
- **96% vs 98% NCII** — different studies/years (2019 Sensity vs 2023 Security Hero); both show NCII dominates, but don't treat them as the same measurement.
- Fraud-loss and percentage-growth figures from security vendors are directional; cite the **named publisher and year**, never a blended number.

---

## Sources (selected)

**Child safety [A]**
- Internet Watch Foundation, *Harm Without Limits: AI CSAM* and 2025 Annual Data & Insights Report (iwf.org.uk) — Jan & May 2026.
- NCMEC CyberTipline Data, 2024 and 2025 (missingkids.org / ncmec.org); Thorn analysis of the 2024 CyberTipline report.
- Stanford Internet Observatory / Cyberlaw — letter on NCMEC AI-CSAM report statistics (the Amazon/78% correction); Bloomberg investigation.
- Crown Prosecution Service & Greater Manchester Police — *R v Hugh Nelson* (Oct 2024).
- NBC News, "The AI child exploitation crisis is here" (Mar 2026).

**Courts / evidence [A/B]**
- Berkeley Technology Law Journal, "Deepfaked Evidence…" (Jun 2025) — Huang v. Tesla, Valenti v. Dfinity, US v. Khalilian, Rittenhouse, US v. Reffitt.
- National Center for State Courts, "AI-generated evidence is a threat to public trust" (Feb 2026) — Mendones v. Cushman & Wakefield.
- CU Boulder Today, deepfakes-in-the-courtroom expert report (Nov 2025).
- NPR, "People are arguing in court that real images are deepfakes" (May 2023) — liar's dividend, Pennypacker ruling.
- Chesney & Citron, "Deep Fakes" (2018) — origin of "liar's dividend."

**Non-consensual / social content [A/B]**
- Ajder et al. / Sensity-Deeptrace (2019) — 96% figure.
- Security Hero / Home Security Heroes, *State of Deepfakes* (2023) — 98% figure.
- Han et al. (2025) — MrDeepFakes scale; Umbach et al., CHI 2024 — 10-country prevalence; Mantzarlis & Lakatos (2025) — nudify-app traffic.
- Vermont Attorney General coalition letter (Aug 2025); FactCheckNI (Feb 2026) — verification of the 96/98/99% claims.

**Scale / fraud [B/C]**
- Deloitte Center for Financial Services (gen-AI fraud forecast); Gartner (2025/2026 surveys); Financial Times/CNN (Arup $25.6M).
- DeepStrike, Keepnet, Sumsub, Resemble AI, Pindrop, iProov — vendor statistics (directional).
- DeepFake-Eval-2024 benchmark; *Computers in Human Behavior Reports* (2024) human-detection meta-analysis.

*Note: figures change quickly and some originate from organisations that sell detection products. Where a number matters for a decision, cite the primary source and its date rather than this summary.*
