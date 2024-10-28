load("@//:python_distribution.bzl", "python_distribution")
python_distribution(
    name = "AUTOAUTO_app",
    packages = [
        "D:/PROJECTS/AUTOAUTO/.venv",
        "D:/PROJECTS/AUTOAUTO/checkpoints",
        "D:/PROJECTS/AUTOAUTO/clip",
        "D:/PROJECTS/AUTOAUTO/DEP",
        "D:/PROJECTS/AUTOAUTO/extensionsbuiltin",
        "D:/PROJECTS/AUTOAUTO/flagged",
        "D:/PROJECTS/AUTOAUTO/models",
        "D:/PROJECTS/AUTOAUTO/out-voice",
        "D:/PROJECTS/AUTOAUTO/output",
        "D:/PROJECTS/AUTOAUTO/roop",
        "D:/PROJECTS/AUTOAUTO/src",
        "D:/PROJECTS/AUTOAUTO/ui",
    ],
    entry_point = "D:/PROJECTS/AUTOAUTO/run:main",  # Define el punto de entrada de tu aplicación
)