'''
import os

def tree_md(folder_path, output_file="output.md"):
    """
    لیست تمام پوشه‌ها و فایل‌ها به شکل درختی و ذخیره در فایل Markdown.
    """
    def tree_lines(current_path, prefix=""):
        """
        تولید خطوط درختی برای مسیر جاری
        """
        items = sorted(os.listdir(current_path))
        lines = []
        for i, item in enumerate(items):
            path = os.path.join(current_path, item)
            connector = "└── " if i == len(items) - 1 else "├── "
            if os.path.isdir(path):
                lines.append(f"{prefix}{connector}**{item}/**")
                # بازگشت بازگشتی برای محتویات پوشه
                extension = "    " if i == len(items) - 1 else "│   "
                lines.extend(tree_lines(path, prefix + extension))
            else:
                lines.append(f"{prefix}{connector}{item}")
        return lines

    # ساخت خطوط درختی
    lines = tree_lines(folder_path)
    
    # نوشتن در فایل Markdown
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ ساختار درختی فایل‌ها در '{output_file}' ذخیره شد.")

# مثال استفاده
if __name__ == "__main__":
    # folder_path = input("مسیر پوشه را وارد کنید: ")
    # tree_md(folder_path)
    # tree_md("e:/BOT-RL-2")
'''




'''
#============================================================================
import os
import fnmatch


def load_gitignore(folder_path):
    """
    خواندن الگوهای .gitignore از پوشه مقصد
    """
    gitignore_path = os.path.join(folder_path, ".gitignore")
    patterns = []

    if os.path.isfile(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    patterns.append(line)

    return patterns


def is_ignored(name, relative_path, patterns):
    """
    بررسی اینکه فایل یا پوشه باید ignore شود یا نه
    """
    for pattern in patterns:
        # تطابق روی نام
        if fnmatch.fnmatch(name, pattern):
            return True
        # تطابق روی مسیر نسبی
        if fnmatch.fnmatch(relative_path, pattern):
            return True
    return False


def tree_md(folder_path, output_file="output.md"):
    """
    تولید نمودار درختی Markdown با رعایت .gitignore
    """
    gitignore_patterns = load_gitignore(folder_path)

    def tree_lines(current_path, prefix=""):
        items = sorted(os.listdir(current_path))
        lines = []

        for i, item in enumerate(items):
            full_path = os.path.join(current_path, item)
            relative_path = os.path.relpath(full_path, folder_path)

            if is_ignored(item, relative_path, gitignore_patterns):
                continue

            connector = "└── " if i == len(items) - 1 else "├── "

            if os.path.isdir(full_path):
                lines.append(f"{prefix}{connector}**{item}/**")
                extension = "    " if i == len(items) - 1 else "│   "
                lines.extend(tree_lines(full_path, prefix + extension))
            else:
                lines.append(f"{prefix}{connector}{item}")

        return lines

    lines = tree_lines(folder_path)

    with open(os.path.join(folder_path, output_file), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ فایل درختی با رعایت .gitignore ساخته شد: {output_file}")


if __name__ == "__main__":
    # folder_path = input("مسیر پوشه را وارد کنید: ")
    # tree_md(folder_path)
    tree_md("e:/BOT-RL-2")
'''

#============================================================================
# f15_testcheck/folder_file_list.py
# Run: python -m f15_testcheck.folder_file_list

import os
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern


def load_gitignore(base_path):
    """
    خواندن .gitignore از پوشه مقصد (اگر وجود داشته باشد)
    """
    gitignore_path = os.path.join(base_path, ".gitignore")
    if not os.path.isfile(gitignore_path):
        return None

    with open(gitignore_path, "r", encoding="utf-8") as f:
        patterns = [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    return PathSpec.from_lines(GitWildMatchPattern, patterns)


def tree_md(base_path, output_file="f15_testcheck/output.md"):
    """
    تولید نمودار درختی Markdown با رعایت .gitignore
    و نمایش پوشه‌ها قبل از فایل‌ها
    """
    spec = load_gitignore(base_path)
    base_path = os.path.abspath(base_path)

    def is_ignored(path):
        if not spec:
            return False
        rel_path = os.path.relpath(path, base_path)
        return spec.match_file(rel_path)

    def walk(current_path, prefix=""):
        entries = os.listdir(current_path)

        # جداسازی پوشه‌ها و فایل‌ها
        dirs = []
        files = []

        for e in entries:
            full_path = os.path.join(current_path, e)
            if is_ignored(full_path):
                continue
            if os.path.isdir(full_path):
                dirs.append(e)
            else:
                files.append(e)

        dirs.sort()
        files.sort()
        items = dirs + files

        lines = []
        for idx, name in enumerate(items):
            full_path = os.path.join(current_path, name)
            is_last = idx == len(items) - 1
            connector = "└── " if is_last else "├── "

            if os.path.isdir(full_path):
                lines.append(f"{prefix}{connector}**{name}/**")
                extension = "    " if is_last else "│   "
                lines.extend(walk(full_path, prefix + extension))
            else:
                lines.append(f"{prefix}{connector}{name}")

        return lines

    lines = walk(base_path)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Tree written to {output_file}")


if __name__ == "__main__":
    # folder_path = input("Path to folder: ").strip()
    # tree_md(folder_path)
    tree_md("e:/Bot-RL-2")