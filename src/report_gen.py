from fpdf import FPDF
import datetime

class LupusReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Quantum-Enhanced Lupus Diagnostic Report', 0, 1, 'C')
        self.ln(10)

def generate_medical_pdf(patient_name, risk_score, symptoms):
    pdf = LupusReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Patient Info
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=1, align='L')
    pdf.cell(200, 10, txt=f"Date: {datetime.date.today()}", ln=1, align='L')
    pdf.ln(5)
    
    # Diagnosis Result
    pdf.set_text_color(255, 0, 0) if risk_score > 70 else pdf.set_text_color(0, 128, 0)
    pdf.cell(200, 10, txt=f"Lupus Risk Probability: {risk_score:.2f}%", ln=1, align='L')
    pdf.set_text_color(0, 0, 0)
    
    # Symptom Summary
    pdf.ln(5)
    pdf.cell(200, 10, txt="Clinical & Genomic Evidence:", ln=1, align='L')
    for symptom, val in symptoms.items():
        pdf.cell(200, 10, txt=f"- {symptom}: {val}", ln=1, align='L')
    
    return pdf.output(dest='S').encode('latin-1')
