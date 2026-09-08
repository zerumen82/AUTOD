@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title AutoAuto - Generador de APK Android (1 clic)
cd /d "%~dp0"

set "BASE=%cd%"
set "TOOLS=%BASE%\android_build_tools"
set "SDK=%TOOLS%\android-sdk"
set "JDK=%TOOLS%\jdk17"
set "GRADLEH=%TOOLS%\gradle-8.7"
set "PROJECT=%BASE%\autoapp"

if not exist "%TOOLS%" mkdir "%TOOLS%"
if exist "%PROJECT%\local.properties" del "%PROJECT%\local.properties" >nul 2>&1

echo ============================================================
echo   AutoAuto APK - Construyendo en 6 pasos (la primera vez
echo   descarga ~2.5 GB de herramientas; luego es instantaneo)
echo ============================================================
echo.

rem ---------------- 1/5 : JDK 17 (Temurin) ----------------
echo [1/5] Preparando Java 17...
if not exist "%JDK%\bin\java.exe" (
    if not exist "%TOOLS%\jdk.zip" (
        echo    Descargando OpenJDK 17 Temurin... ^(en curso^)
        powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.12%%2B7/OpenJDK17U-jdk_x64_windows_hotspot_17.0.12_7.zip' -OutFile '%TOOLS%\jdk.zip'"
    )
    echo    Extrayendo JDK...
    powershell -NoProfile -Command "Expand-Archive -Path '%TOOLS%\jdk.zip' -DestinationPath '%TOOLS%' -Force"
    for /d %%d in ("%TOOLS%\jdk-17*") do ren "%%d" "jdk17" 2>nul
)
set "JAVA_HOME=%JDK%"
if not exist "%JAVA_HOME%\bin\java.exe" for /d %%d in ("%TOOLS%\jdk-17*") do set "JAVA_HOME=%%~d"
set "PATH=%JAVA_HOME%\bin;%PATH%"
"%JAVA_HOME%\bin\java.exe" -version >nul 2>&1 || (echo [ERROR] No se pudo preparar Java 17. & pause & exit /b 1)

rem ---------------- 2/5 : Gradle 8.7 ----------------
echo [2/5] Preparando Gradle...
if not exist "%GRADLEH%\bin\gradle.bat" (
    if not exist "%TOOLS%\gradle.zip" (
        echo    Descargando Gradle 8.7...
        powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://services.gradle.org/distributions/gradle-8.7-bin.zip' -OutFile '%TOOLS%\gradle.zip'"
    )
    echo    Extrayendo Gradle...
    powershell -NoProfile -Command "Expand-Archive -Path '%TOOLS%\gradle.zip' -DestinationPath '%TOOLS%' -Force"
)

rem ---------------- 3/5 : Android SDK ----------------
echo [3/5] Preparando Android SDK (comando-linea)...
if not exist "%SDK%\cmdline-tools\latest\bin\sdkmanager.bat" (
    if not exist "%TOOLS%\cmdtools.zip" (
        echo    Descargando Android command-line tools...
        powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://dl.google.com/android/repository/commandlinetools-win-11076708_latest.zip' -OutFile '%TOOLS%\cmdtools.zip'"
    )
    echo    Instalando SDK...
    powershell -NoProfile -Command "Expand-Archive -Path '%TOOLS%\cmdtools.zip' -DestinationPath '%TOOLS%\cmdtools_extract' -Force"
    if exist "%SDK%\cmdline-tools\latest" rmdir /s /q "%SDK%\cmdline-tools\latest"
    mkdir "%SDK%\cmdline-tools\latest"
    move /y "%TOOLS%\cmdtools_extract\cmdline-tools\*" "%SDK%\cmdline-tools\latest\" >nul
    rmdir /s /q "%TOOLS%\cmdtools_extract" >nul 2>&1
)
set "ANDROID_HOME=%SDK%"
set "ANDROID_SDK_ROOT=%SDK%"

rem ---------------- 4/5 : Licencias + plataforma 34 ----------------
echo [4/5] Aceptando licencias y descargando plataforma Android 34...
if not exist "%SDK%\platforms\android-34" (
    echo y| call "%SDK%\cmdline-tools\latest\bin\sdkmanager.bat" --sdk_root="%SDK%" --licenses >nul 2>&1
    call "%SDK%\cmdline-tools\latest\bin\sdkmanager.bat" --sdk_root="%SDK%" "platform-tools" "platforms;android-34" "build-tools;34.0.0" >nul 2>&1
)

rem Escribir local.properties apuntando al SDK
> "%PROJECT%\local.properties" echo sdk.dir=%SDK:\=/%

rem ---------------- 5/6 : Crear clave de firma (solo la primera vez) ----------------
set "KS=%PROJECT%\release-keystore.jks"
set "KSPROP=%PROJECT%\release-keystore.properties"
if not exist "%KSPROP%" (
    echo [5/6] Generando clave de firma para el APK release...
    "%JAVA_HOME%\bin\keytool.exe" -genkeypair -v -keystore "%KS%" -storepass autoauto123 -alias autoauto -keypass autoauto123 -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=AutoAuto, OU=Dev, O=AutoAuto, L=City, ST=State, C=ES" >nul 2>&1
    > "%KSPROP%" echo storeFile=%KS:\=\\%
    >> "%KSPROP%" echo storePassword=autoauto123
    >> "%KSPROP%" echo keyAlias=autoauto
    >> "%KSPROP%" echo keyPassword=autoauto123
    echo    Clave creada.
) else (
    echo [5/6] Clave de firma existente detectada.
)

rem ---------------- 6/6 : Compilar APK firmado ----------------
echo [6/6] Compilando APK release firmado (primera vez tarda, descarga dependencias)...
cd /d "%PROJECT%"
call "%GRADLEH%\bin\gradle.bat" -p "%PROJECT%" assembleRelease --no-daemon
if errorlevel 1 (
    echo.
    echo [ERROR] La compilacion fallo. Revisa el mensaje de arriba.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   LISTO! Tu APK (firmado, sin "origen desconocido") esta en:
echo   %PROJECT%\app\build\outputs\apk\release\app-release.apk
echo.
echo   Copiala a tu Android e instala. Al abrirla se conecta sola
echo   a tu PC si AutoAuto esta encendido (misma red WiFi, 7861).
echo ============================================================
pause
