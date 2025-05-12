# Manual Comparison Tool

To facilitate manual correction of hallucinations in model generated summaries of reports, we develop a lightweight, interactive comparison tool. 
The tool is implemented in pure HTML and JavaScript and runs locally without a need for an internet connection.

This tool allows users to copy/paste the original and summarized reports into two text boxes. Users then select text from one box which gets automatically highlighted in the other. This enables users to quickly identify the source of individual components of the summary or vice versa. This is especially helpful when the original report is much longer than the summary and would otherwise require a long time to scan through the entire document. Additionally, as the user selections are dynamic, information extracted around corrected typos are easier to identify by subselecting words or phrases.

To use the tool, simply [download the file](compare.html) and open it in your preferred web browser.

A brief demo video of its usage is available on Google Drive: https://drive.google.com/file/d/1bqchsRuhn2FOVn7slPQroMxIPJ1yd2lB/view?usp=sharing

### Our Experiments

Here, we document the manual correction procedure we adopt for our experiments, however the usage of the tool is not limited to this procedure.

> In our manual correction of generated summaries, we only change factually incorrect information based on information from the original report. We do not add extra information that was not already present in the summary. When the incorrect information cannot be corrected based on the original report, we delete the erroneous text. A salient example of this was when patient age was redacted in the original report. The resulting extracted text thus contained a fragment such as "-year-old patient", which the summarizing LLM interpreted to mean a 1-year-old patient. Manual verification of the case metadata revealed the patient to be in their 40s, however, because this data was impossible to derive from the original report, we remove the mention of the patient age in the corrected summary. All manual corrections for our experiment were done by a medical student who had completed two years of preclinical medical education.