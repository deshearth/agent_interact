"""Check the report artifacts against their content and saved source evidence."""
from pathlib import Path
from html import unescape
from zipfile import ZipFile
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import pdfplumber
from pypdf import PdfReader
import render_report as renderer

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
CONTENT = json.loads((OUT / 'report_content.json').read_text())
EVIDENCE = json.loads((OUT / 'evidence_index.json').read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def normalized(text):
    return re.sub(r'\s+', '', unescape(re.sub(r'<[^>]+>', '', str(text))))

def source(path):
    return json.loads((ROOT / path).read_text())

def table(prefix):
    return next(b for p in CONTENT['pages'] for b in p['blocks']
                if b['type'] == 'table' and b['caption'].startswith(prefix))

def main():
    checks = {}
    mismatches = []
    unique_sources = {}
    for group in EVIDENCE['evidence']:
        for entry in group['files']:
            actual = digest(ROOT / entry['path'])
            unique_sources[entry['path']] = actual
            if actual != entry['sha256']:
                mismatches.append(entry['path'])
    assert not mismatches, mismatches
    checks['evidence_groups'] = len(EVIDENCE['evidence'])
    checks['source_hashes_verified'] = len(unique_sources)
    template = Path('/Users/melancholymirth/Downloads') / EVIDENCE['template_name']
    assert digest(template) == EVIDENCE['template_sha256']
    checks['template_hash_verified'] = True

    sweep = source('runs/breadth_320_size_sweep_v1/results/summary.json')
    expected = []
    for key in ['I0', 'pico', 'nano', 'tiny', 'small', 'full', 'wide']:
        scores = sweep['summary'][key]
        expected.append(['Ising' if key == 'I0' else 'Transformer',
                         f"{sweep['parameters'][key]:,}",
                         f"{100*(1-scores['train']['error']):.2f}%",
                         f"{100*(1-scores['validation']['error']):.2f}%",
                         f"{scores['validation']['nll']:.5f}"])
    assert table('Table 2.')['rows'] == expected
    checks['main_size_table_source_match'] = True

    curve = source('runs/breadth_320_learning_curve_v1/results/summary.json')
    expected = []
    for count in [40, 80, 160, 320]:
        row = [count]
        for key in ['I0', 'nano', 'tiny', 'wide']:
            v = curve['summary'][str(count)][key]['validation']
            row.append(f"{100*(1-v['error']):.2f}% / {v['nll']:.5f}")
        expected.append(row)
    assert [[str(x) for x in row] for row in table('Table F1.')['rows']] == [[str(x) for x in row] for row in expected]
    checks['learning_curve_table_source_match'] = True
    assert not sweep['assessment_loaded'] and not curve['assessment_loaded']
    checks['saved_primary_studies_assessment_unloaded'] = True

    missing = sorted({c for c in json.dumps(CONTENT, ensure_ascii=False)
                      if ord(c) > 127 and ord(c) not in renderer.pdfmetrics.getFont('Body').face.charToGlyph})
    assert not missing, missing
    checks['unsupported_report_font_glyphs'] = missing
    checks['pdfs'] = {}
    for label, path in [('report', OUT/'agent_interaction_report.pdf'),
                        ('word_render', OUT/'qa/word_verified/agent_interaction_report.pdf')]:
        reader = PdfReader(path)
        texts = [p.extract_text() for p in reader.pages]
        assert len(texts) == 14
        assert 'References' in texts[4]
        for section in ['1. Introduction', '2. Related Work', '3. Methods', '4. Results', '5. Discussion and Limitations', '6. Conclusion']:
            assert section in '\n'.join(texts[:4]), section
        assert all(len(t.split()) > 200 for t in texts)
        out_of_bounds = []
        with pdfplumber.open(path) as pdf:
            for n, page in enumerate(pdf.pages, 1):
                for char in page.chars:
                    if char['x0'] < -0.5 or char['x1'] > page.width+0.5 or char['top'] < -0.5 or char['bottom'] > page.height+0.5:
                        out_of_bounds.append({'page': n, 'text': char['text']})
        assert not out_of_bounds, out_of_bounds[:10]
        checks['pdfs'][label] = {'pages': len(texts), 'references_page': 5,
                                'words_by_page': [len(t.split()) for t in texts],
                                'out_of_page_characters': 0}

    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
          'dc': 'http://purl.org/dc/elements/1.1/'}
    with ZipFile(OUT/'agent_interaction_report.docx') as z:
        document = ET.fromstring(z.read('word/document.xml'))
        doc_text = normalized(''.join(e.text or '' for e in document.iter(f"{{{ns['w']}}}t")))
        missing_blocks = []
        checked = 0
        for p in CONTENT['pages']:
            for b in p['blocks']:
                strings = [b['text']] if 'text' in b else []
                if 'subtitle' in b: strings.append(b['subtitle'])
                if 'caption' in b: strings.append(b['caption'])
                if b['type'] == 'table':
                    strings.extend(b['headers'])
                    strings.extend(x for row in b['rows'] for x in row)
                for s in strings:
                    checked += 1
                    if normalized(s) not in doc_text:
                        missing_blocks.append(str(s)[:100])
        assert not missing_blocks, missing_blocks
        images = [n for n in z.namelist() if n.startswith('word/media/')]
        assert len(images) == 2
        creator = ET.fromstring(z.read('docProps/core.xml')).find('dc:creator', ns)
        assert creator is None or not creator.text
        title_paragraphs = [
            p for p in document.findall('.//w:p', ns)
            if p.find('w:pPr/w:pStyle', ns) is not None and p.find('w:pPr/w:pStyle', ns).get(f"{{{ns['w']}}}val") == 'Title']
        assert len(title_paragraphs) == 1
        checks['docx'] = {'content_fragments_verified': checked, 'embedded_images': len(images),
                          'author_metadata_empty': True, 'title_style_present': True,
                          'hyperlinks': len(document.findall('.//w:hyperlink', ns))}

    assert CONTENT['authors_omitted_by_request']
    assert 150 <= CONTENT['abstract_word_count'] <= 250
    checks['abstract_words'] = CONTENT['abstract_word_count']
    checks['authors_omitted'] = True
    checks['source_generation_for_report'] = EVIDENCE['source_calls_for_report']
    checks['new_training_for_report'] = EVIDENCE['new_training_for_report']
    outputs = ['agent_interaction_report.pdf', 'agent_interaction_report.docx',
               'report_content.json', 'evidence_index.json', 'prepare_report.py',
               'render_report.py', 'verify_report.py', 'figures/pipeline.png',
               'figures/learning_curve_main.png']
    checks['artifact_sha256'] = {name: digest(OUT/name) for name in outputs}
    checks['status'] = 'passed'
    (OUT/'qa/artifact_audit.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2))

if __name__ == '__main__':
    main()
