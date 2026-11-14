"""Render the frozen report content to PDF and an editable Word document."""
from pathlib import Path
from html import unescape
from html.parser import HTMLParser
import hashlib
import json
import re

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib.utils import ImageReader
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE

OUT=Path(__file__).resolve().parent
CONTENT=json.loads((OUT/'report_content.json').read_text())
FONT=Path('/System/Library/Fonts/Supplemental')
for name,file in [('Body','Times New Roman.ttf'),('Body-Bold','Times New Roman Bold.ttf'),('Body-Italic','Times New Roman Italic.ttf'),('Body-BoldItalic','Times New Roman Bold Italic.ttf'),('Sans','Arial.ttf'),('Sans-Bold','Arial Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONT/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Body-Bold',italic='Body-Italic',boldItalic='Body-BoldItalic')
pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='Sans-Bold',italic='Sans',boldItalic='Sans-Bold')
pdfmetrics.registerFont(TTFont('Mono',str(FONT/'Courier New.ttf')))

W,H=612,792
MARGIN=62
TEXTW=W-2*MARGIN
styles={
 'p':ParagraphStyle('p',fontName='Body',fontSize=10.5,leading=12.6,alignment=TA_JUSTIFY,spaceAfter=6),
 'h1':ParagraphStyle('h1',fontName='Body-Bold',fontSize=13.5,leading=16,spaceBefore=7,spaceAfter=6,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='Body-Bold',fontSize=11,leading=13.2,spaceBefore=5,spaceAfter=4,keepWithNext=True),
 'title':ParagraphStyle('title',fontName='Body-Bold',fontSize=20,leading=23,alignment=TA_CENTER,spaceAfter=5),
 'subtitle':ParagraphStyle('subtitle',fontName='Body',fontSize=12,leading=14,alignment=TA_CENTER,spaceAfter=8),
 'date':ParagraphStyle('date',fontName='Body',fontSize=9,leading=11,alignment=TA_CENTER,spaceAfter=11),
 'caption':ParagraphStyle('caption',fontName='Body',fontSize=9,leading=10.6,alignment=TA_LEFT,spaceAfter=5,spaceBefore=3),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=9,leading=10.5,alignment=TA_LEFT),
 'th':ParagraphStyle('th',fontName='Body-Bold',fontSize=9,leading=10.5,alignment=TA_LEFT),
}

def clean(t):
    return str(t).replace('&','&amp;').replace('×','×') if '<' not in str(t) else str(t)

def footer(c,doc):
    c.saveState()
    if doc.page>1:
        c.setFont('Body',8);c.setFillColor(colors.HexColor('#565e63'))
        c.drawString(MARGIN,H-32,'COMPACT MODELS OF LLM AGENT INTERACTION')
        c.setLineWidth(.4);c.line(MARGIN,H-39,W-MARGIN,H-39)
    else:
        c.setLineWidth(1.7);c.line(MARGIN,H-42,W-MARGIN,H-42)
    c.setLineWidth(.4);c.setStrokeColor(colors.HexColor('#8b9195'));c.line(MARGIN,42,W-MARGIN,42)
    c.setFillColor(colors.HexColor('#565e63'));c.setFont('Body',8)
    c.drawString(MARGIN,29,'Research report · 14 November 2026')
    c.drawRightString(W-MARGIN,29,str(doc.page))
    c.restoreState()

class ReportDoc(BaseDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='h1':
            txt=re.sub('<[^>]+>','',flowable.getPlainText())
            key='section-'+str(len(self.outline))
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(txt,key,0,False)
            self.outline.append({'title':txt,'page':self.page})

def render_pdf():
    path=OUT/'agent_interaction_report.pdf'
    doc=ReportDoc(str(path),pagesize=(W,H),leftMargin=MARGIN,rightMargin=MARGIN,topMargin=54,bottomMargin=51,title=CONTENT['title'],author='',subject='Apart Research template; project evidence synthesis')
    doc.outline=[]
    doc.addPageTemplates([PageTemplate(id='normal',frames=[Frame(MARGIN,51,TEXTW,H-105,leftPadding=0,rightPadding=0,bottomPadding=0,topPadding=0)],onPage=footer)])
    story=[]
    for i,page in enumerate(CONTENT['pages']):
        if i:story.append(PageBreak())
        for b in page['blocks']:
            typ=b['type']
            if typ=='title':
                story.extend([Spacer(1,4),Paragraph(b['text'],styles['title']),Paragraph(b['subtitle'],styles['subtitle'])])
            elif typ in ['p','h1','h2','date']:
                story.append(Paragraph(b['text'],styles[typ]))
            elif typ=='figure':
                file=OUT/'figures'/b['file'];iw,ih=ImageReader(str(file)).getSize();hgt=min(b['height'],TEXTW*ih/iw)
                picture=Image(str(file),width=TEXTW,height=hgt)
                story.append(KeepTogether([picture,Paragraph(b['caption'],styles['caption']),Spacer(1,3)]))
            elif typ=='table':
                rows=[[Paragraph(clean(x),styles['th' if j==0 else 'cell']) for x in row] for j,row in enumerate([b['headers']]+b['rows'])]
                tab=Table(rows,colWidths=[TEXTW*x for x in b['widths']],repeatRows=1,hAlign='LEFT')
                tab.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef0f2')),('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.6,colors.black),('LINEBELOW',(0,1),(-1,-2),.25,colors.HexColor('#d5dade'))]))
                story.append(KeepTogether([Paragraph(b['caption'],styles['caption']),tab,Spacer(1,7)]))
    doc.build(story)
    return doc.outline

class WordInline(HTMLParser):
    def __init__(self,paragraph):
        super().__init__();self.p=paragraph;self.bold=0;self.italic=0;self.mono=0;self.href=None;self.sub=0;self.sup=0
    def handle_starttag(self,tag,attrs):
        if tag=='b':self.bold+=1
        if tag=='i':self.italic+=1
        if tag=='font':self.mono+=1
        if tag=='sub':self.sub+=1
        if tag in ('sup','super'):self.sup+=1
        if tag=='link':self.href=dict(attrs).get('href')
        if tag=='br':self.p.add_run().add_break()
    def handle_endtag(self,tag):
        if tag=='b':self.bold-=1
        if tag=='i':self.italic-=1
        if tag=='font':self.mono-=1
        if tag=='sub':self.sub-=1
        if tag in ('sup','super'):self.sup-=1
        if tag=='link':self.href=None
    def handle_data(self,data):
        r=self.p.add_run(unescape(data));r.bold=True if self.bold else None;r.italic=True if self.italic else None
        if self.mono:r.font.name='Courier New';r.font.size=Pt(8.5)
        if self.sub:r.font.subscript=True
        if self.sup:r.font.superscript=True
        if self.href:
            hyperlink=OxmlElement('w:hyperlink')
            hyperlink.set(qn('r:id'),self.p.part.relate_to(self.href,RELATIONSHIP_TYPE.HYPERLINK,is_external=True))
            hyperlink.append(r._r);self.p._p.append(hyperlink)

def inline(paragraph,text):WordInline(paragraph).feed(str(text))

def render_docx():
    d=Document();sec=d.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.left_margin=sec.right_margin=Pt(MARGIN);sec.top_margin=Pt(54);sec.bottom_margin=Pt(51)
    sec.header_distance=Pt(25);sec.footer_distance=Pt(25)
    normal=d.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10.5)
    normal.paragraph_format.line_spacing=Pt(12.6);normal.paragraph_format.space_after=Pt(4)
    title_style=d.styles['Title'];title_style.font.name='Times New Roman';title_style.font.size=Pt(20)
    title_style.font.bold=True;title_style.font.color.rgb=RGBColor(0,0,0)
    title_style.paragraph_format.line_spacing=Pt(23);title_style.paragraph_format.space_after=Pt(5)
    for border in list(title_style.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
    for name,size in [('Heading 1',13.5),('Heading 2',11)]:
        s=d.styles[name];s.font.name='Times New Roman';s.font.size=Pt(size);s.font.color.rgb=RGBColor(0,0,0)
        s.paragraph_format.space_before=Pt(7 if name=='Heading 1' else 5);s.paragraph_format.space_after=Pt(6 if name=='Heading 1' else 4)
    header=sec.header.paragraphs[0];header.text='COMPACT MODELS OF LLM AGENT INTERACTION';header.runs[0].font.size=Pt(8)
    foot=sec.footer.paragraphs[0];foot.text='Research report · 14 November 2026';foot.runs[0].font.size=Pt(8)
    foot.add_run(' '*8);field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');foot._p.append(field)
    for i,page in enumerate(CONTENT['pages']):
        for block_index,b in enumerate(page['blocks']):
            typ=b['type']
            if typ=='title':
                q=d.add_paragraph(style='Title');q.alignment=WD_ALIGN_PARAGRAPH.CENTER;r=q.add_run(b['text']);r.bold=True;r.font.size=Pt(20)
                q=d.add_paragraph(b['subtitle']);q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.runs[0].font.size=Pt(12)
            elif typ=='date':
                q=d.add_paragraph(b['text']);q.alignment=WD_ALIGN_PARAGRAPH.CENTER;q.runs[0].font.size=Pt(9)
            elif typ in ['p','h1','h2']:
                q=d.add_paragraph(style='Heading 1' if typ=='h1' else 'Heading 2' if typ=='h2' else 'Normal');inline(q,b['text'])
                if typ=='p':q.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
                if i and block_index==0:q.paragraph_format.page_break_before=True
            elif typ=='figure':
                file=OUT/'figures'/b['file'];iw,ih=ImageReader(str(file)).getSize();hgt=min(b['height'],TEXTW*ih/iw)
                q=d.add_paragraph();q.paragraph_format.keep_with_next=True;q.paragraph_format.line_spacing=1.0
                q.add_run().add_picture(str(file),width=Pt(TEXTW),height=Pt(hgt))
                q=d.add_paragraph(b['caption']);q.paragraph_format.space_after=Pt(7)
                q.paragraph_format.line_spacing=Pt(10.6)
                for r in q.runs:r.font.size=Pt(9)
            elif typ=='table':
                q=d.add_paragraph(b['caption']);q.paragraph_format.keep_with_next=True
                q.paragraph_format.line_spacing=Pt(10.6);q.paragraph_format.space_after=Pt(5)
                for r in q.runs:r.font.size=Pt(9)
                t=d.add_table(rows=1,cols=len(b['headers']));t.autofit=False
                for j,w in enumerate(b['widths']):t.columns[j].width=Pt(TEXTW*w)
                for j,row in enumerate([b['headers']]+b['rows']):
                    cells=t.rows[0].cells if j==0 else t.add_row().cells
                    trpr=cells[0]._tc.getparent().get_or_add_trPr();cant=OxmlElement('w:cantSplit');trpr.append(cant)
                    if j==0:
                        repeat=OxmlElement('w:tblHeader');trpr.append(repeat)
                    for k,x in enumerate(row):
                        cell=cells[k];cell.width=Pt(TEXTW*b['widths'][k]);q=cell.paragraphs[0];q.paragraph_format.line_spacing=Pt(10.5);q.paragraph_format.space_after=Pt(3);q.paragraph_format.space_before=Pt(3)
                        inline(q,str(x))
                        for r in q.runs:r.font.size=Pt(9);r.bold=j==0
                        if j==0:
                            sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'EEF0F2');cell._tc.get_or_add_tcPr().append(sh)
                spacer=d.add_paragraph();spacer.paragraph_format.line_spacing=Pt(1);spacer.paragraph_format.space_after=Pt(6)
                spacer.add_run().font.size=Pt(1)
    d.core_properties.title=CONTENT['title'];d.core_properties.author='';d.core_properties.subject='Project report using the Apart Research submission structure'
    d.save(OUT/'agent_interaction_report.docx')

def render_preview():
    import pypdfium2 as pdfium
    from pypdf import PdfReader
    from PIL import Image as PILImage,ImageOps,ImageDraw
    pdf=pdfium.PdfDocument(str(OUT/'agent_interaction_report.pdf'))
    reader=PdfReader(OUT/'agent_interaction_report.pdf')
    for stale in (OUT/'qa').glob('page_[0-9][0-9].png'):
        if int(stale.stem.split('_')[-1]) > len(pdf):stale.unlink()
    texts=[];thumbs=[]
    for i,page in enumerate(pdf):
        im=page.render(scale=1.5).to_pil().convert('RGB');im.save(OUT/'qa'/f'page_{i+1:02d}.png')
        thumb=im.copy();thumb.thumbnail((306,396));thumbs.append(thumb)
        texts.append(reader.pages[i].extract_text())
    cols=3;rows=(len(thumbs)+cols-1)//cols
    contact=PILImage.new('RGB',(cols*326,rows*426),'#dedede');draw=ImageDraw.Draw(contact)
    for i,im in enumerate(thumbs):
        x=(i%cols)*326+10;y=(i//cols)*426+20;contact.paste(im,(x,y));draw.text((x,y-15),f'Page {i+1}',fill='black')
    contact.save(OUT/'qa/contact_sheet.png')
    (OUT/'qa/extracted_text.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(texts)))
    return [{'page':i+1,'words':len(t.split()),'start':t[:120],'end':t[-100:]} for i,t in enumerate(texts)]

if __name__=='__main__':
    outline=render_pdf();render_docx();pages=render_preview()
    record={'pages':pages,'outline':outline,'abstract_words':CONTENT['abstract_word_count'],'authors_omitted':True}
    (OUT/'qa/layout.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))
