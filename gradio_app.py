import gradio as gr
from final_pipeline import main

gr.Interface(fn=main,inputs=gr.inputs.Image(shape=(512, 512)),
             outputs=[gr.outputs.Image(type = 'numpy'), gr.outputs.Image(type = 'numpy')],
title='Scene Graph Generation', interpretation='default',enable_queue=True).launch(share=True)
