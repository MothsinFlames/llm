import fitz  # PyMuPDF
import re

def pdf_to_markdown(pdf_path, md_path):
    doc = fitz.open(pdf_path)
    markdown_lines = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:  # 文本块
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if not text:
                            continue
                        # 根据字体加粗/斜体判断 Markdown 样式
                        font = span["font"]
                        size = span["size"]
                        flags = span["flags"]

                        # 简单判断加粗（flag 16 = 加粗）
                        is_bold = flags & 16
                        is_italic = flags & 2

                        if is_bold and is_italic:
                            text = f"***{text}***"
                        elif is_bold:
                            text = f"**{text}**"
                        elif is_italic:
                            text = f"*{text}*"

                        line_text += text + " "

                    # 去掉多余空格
                    line_text = line_text.strip()
                    if line_text:
                        markdown_lines.append(line_text)

            elif "image" in block:  # 图片（可选处理）
                # 可以保存图片并插入 Markdown 图片语法
                continue

        # 每页之间加一个分页符（可选）
        markdown_lines.append("\n---\n")

    # 写入 Markdown 文件
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))

    print(f"✅ 转换完成！保存为：{md_path}")

# 使用示例
pdf_to_markdown(r"d:\rag2025\0714案例库_files\pdf\昇腾维护问题处理指导\[X板斧]昇腾现网网口闪断类问题处理三板斧-V1.0\昇腾计算维护现网网口闪断类问题处理三板斧—V1.0.pdf", r"C:\Users\w30057236\Desktop\1\output.md")