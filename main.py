import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import tkinter as tk
from tkinter import scrolledtext, Toplevel, Text
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import io
import os
from textwrap import wrap

# Configuración de modelos y tokenizers
question_generator = pipeline("text-generation", model="distilgpt2")
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate_questions(job_description, num_questions=5):
    prompt = f"Generate {num_questions} interview questions for the following job description: {job_description}\n\nInterview Questions:\n"
    generated_text = question_generator(prompt, max_length=100*num_questions, num_return_sequences=1)[0]['generated_text']
    questions = generated_text.split('\n')
    questions = [q.strip() for q in questions if q.strip() and q.strip().endswith('?')]
    return questions[:num_questions]

def evaluate_response(response):
    inputs = tokenizer(response, return_tensors="pt")
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    score = scores[0][1].item()  # La puntuación de la clase positiva (buena respuesta)
    return score

def start_interview():
    global questions, total_score, question_count, current_question, responses, scores
    job_description = job_description_entry.get("1.0", tk.END).strip()
    questions = generate_questions(job_description)
    total_score = 0
    question_count = 0
    current_question = 0
    responses = []
    scores = []
    job_description_entry.config(state=tk.DISABLED)  # Desactivar la caja de texto de descripción del trabajo
    generate_button.config(state=tk.DISABLED)  # Desactivar el botón de Generate Questions
    submit_button.config(state=tk.NORMAL)  # Activar el botón de Submit Response
    show_next_question()

def show_next_question():
    global current_question
    if current_question < len(questions):
        questions_text.config(state=tk.NORMAL)
        questions_text.delete("1.0", tk.END)
        questions_text.insert(tk.END, f"Question {current_question + 1}: {questions[current_question]}\n")
        questions_text.config(state=tk.DISABLED)
        response_entry.delete("1.0", tk.END)
    else:
        global_score = total_score / question_count if question_count > 0 else 0
        global_score_label.config(text=f"Global Score: {global_score:.2f}")
        result_label.config(text="Interview Completed.")
        job_description_entry.config(state=tk.NORMAL)  # Habilitar la caja de texto de descripción del trabajo
        generate_button.config(state=tk.NORMAL)  # Habilitar el botón de Generate Questions
        submit_button.config(state=tk.DISABLED)  # Desactivar el botón de Submit Response
        questions_text.config(state=tk.NORMAL)
        questions_text.delete("1.0", tk.END)  # Limpiar la caja de texto de preguntas
        questions_text.config(state=tk.DISABLED)
        response_entry.delete("1.0", tk.END)  # Limpiar la caja de texto de respuestas
        save_pdf_button.config(state=tk.NORMAL)  # Activar el botón de Save PDF

def submit_response():
    global current_question, total_score, question_count, responses, scores
    response = response_entry.get("1.0", tk.END).strip()
    if response:  # Asegúrate de que hay una respuesta antes de evaluarla
        score = evaluate_response(response)
        result_label.config(text=f"Response Score: {score:.2f}")
        total_score += score
        question_count += 1
        current_question += 1
        responses.append(response)
        scores.append(score)
        response_entry.delete("1.0", tk.END)  # Limpiar la caja de respuesta
        show_next_question()

def wrap_text(text, max_width):
    return "<br />".join(wrap(text, width=max_width))

def save_pdf():
    global questions, responses, scores, total_score, question_count
    pdf_file = "interview_results.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Título
    title = Paragraph("Interview Results", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Puntaje Global
    global_score = total_score / question_count if question_count > 0 else 0
    score_text = Paragraph(f"Global Score: {global_score:.2f}", styles['Heading2'])
    elements.append(score_text)
    elements.append(Spacer(1, 12))

    # Tabla de preguntas y respuestas
    table_data = [["Question", "Response", "Score"]]
    for i, (q, r, s) in enumerate(zip(questions, responses, scores)):
        wrapped_q = Paragraph(wrap_text(q, 60), styles['Normal'])
        wrapped_r = Paragraph(wrap_text(r, 60), styles['Normal'])
        table_data.append([wrapped_q, wrapped_r, f"{s:.2f}"])

    table = Table(table_data, colWidths=[2 * inch, 3.5 * inch, 1 * inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 24))

    # Crear y guardar la gráfica
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, question_count + 1), scores, marker='o', color='blue', linestyle='-', linewidth=2, markersize=5)
    ax.set_xlabel('Question Number')
    ax.set_ylabel('Score')
    ax.setTitle('Interview Performance')

    # Guardar la gráfica temporalmente
    graph_file = "performance_plot.png"
    plt.savefig(graph_file)
    plt.close(fig)  # Cerrar la figura

    # Agregar la gráfica al PDF
    from reportlab.platypus import Image
    elements.append(Image(graph_file, width=6 * inch, height=3 * inch))
    
    # Construir el documento
    doc.build(elements)

    # Eliminar la imagen temporal
    os.remove(graph_file)

    result_label.config(text="PDF Saved!")

def show_help():
    help_window = Toplevel(root)
    help_window.title("User Manual")
    help_window.geometry("400x300")

    help_text = Text(help_window, wrap="word")
    help_text.insert(tk.END, """
    Welcome to AiInterview: Interview Simulator!

    How to Use:

    1. Enter the job description in the 'Job Description' box.
    2. Click on 'Generate Questions' to create interview questions based on the job description.
    3. Answer the questions in the 'Your Response' box and click 'Submit Response'.
    4. After answering all questions, you can save the results as a PDF by clicking 'Save PDF'.
    5. Use the 'Help' button anytime to view this manual.

    Tips:
    - Make sure your job description is detailed for better question generation.
    - Provide thoughtful responses to get accurate evaluations.

    Enjoy your practice interview!
    """)
    help_text.config(state=tk.DISABLED)
    help_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Variables globales
questions = []
total_score = 0
question_count = 0
current_question = 0
responses = []
scores = []

# Configuración de la interfaz gráfica
root = tk.Tk()
root.title("AiInterview: Interview Simulator")

tk.Label(root, text="Job Description:").pack(fill=tk.X, padx=10, pady=5)
job_description_entry = scrolledtext.ScrolledText(root, height=5, width=50)
job_description_entry.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

generate_button = tk.Button(root, text="Generate Questions", command=start_interview)
generate_button.pack(fill=tk.X, padx=10, pady=5)

tk.Label(root, text="Questions:").pack(fill=tk.X, padx=10, pady=5)
questions_text = scrolledtext.ScrolledText(root, height=10, width=50, state=tk.DISABLED)
questions_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

tk.Label(root, text="Your Response:").pack(fill=tk.X, padx=10, pady=5)
response_entry = scrolledtext.ScrolledText(root, height=5, width=50)
response_entry.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

submit_button = tk.Button(root, text="Submit Response", command=submit_response, state=tk.DISABLED)
submit_button.pack(fill=tk.X, padx=10, pady=5)

save_pdf_button = tk.Button(root, text="Save PDF", command=save_pdf, state=tk.DISABLED)
save_pdf_button.pack(fill=tk.X, padx=10, pady=5)

help_button = tk.Button(root, text="Help", command=show_help)
help_button.pack(fill=tk.X, padx=10, pady=5)

result_label = tk.Label(root, text="Response Score: ")
result_label.pack(fill=tk.X, padx=10, pady=5)

global_score_label = tk.Label(root, text="Global Score: ")
global_score_label.pack(fill=tk.X, padx=10, pady=5)

root.mainloop()
