@echo off
echo Copying files to wix\package folder...

if not exist package mkdir package

REM Copy flm.exe from build folder
echo Copying flm.exe...
copy "..\build\flm.exe" "package\flm.exe"

REM Copy all DLL files from lib folder
echo Copying DLL files...
copy "..\lib\*.dll" "package\"
copy "..\lib\xrt\*.dll" "package\"

REM Copy model_list.json and model_info.json from root
echo Copying model_list.json...
copy "..\model_list.json" "package\model_list.json"
copy "..\model_info.json" "package\model_info.json"

REM Copy third-party license notices from repo root
echo Copying THIRD_PARTY_NOTICES.txt...
copy "..\..\THIRD_PARTY_NOTICES.txt" "package\THIRD_PARTY_NOTICES.txt"

REM Stage the model-extension SDK from the local build unless CI staged it
if not exist "package\sdk" (
    echo Installing the model-extension SDK...
    cmake --install "..\build" --prefix "package\sdk" --component ModelExtension --config Release
    if errorlevel 1 exit /b 1
)

REM Copy static installer assets
echo Copying static assets...
copy "..\inno\logo.ico" "package\logo.ico"
copy "..\inno\terms.rtf" "package\terms.rtf"

echo Done!
