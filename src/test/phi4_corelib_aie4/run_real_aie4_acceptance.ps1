param(
  [string]$FlmExe = 'src/build-aie4/Release/flm.exe',
  [string]$Model = 'phi4-mini-it-aie4:4b',
  [string]$CorelibDll = 'C:/Users/chiz/work/ryzenai-corelib/install/bin/ryzenai_corelib.dll',
  [string]$Output = 'src/build-aie4/phi4-gguf-aie4-acceptance.json',
  [int]$Port = 52625,
  [string]$Python = 'python',
  # Diagnostics only: skips the 16-minute CLI matrix so the REST phase can be
  # iterated on quickly. A record produced this way can never report success.
  [switch]$SkipCli
)
$ErrorActionPreference='Stop'
$root=(Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$exe=(Resolve-Path (Join-Path $root $FlmExe)).Path
$core=(Resolve-Path $CorelibDll).Path
$outPath=[IO.Path]::GetFullPath((Join-Path $root $Output))
$outDir=Split-Path $outPath
$modelDir=Join-Path $env:USERPROFILE '.flm/models/phi4-mini-it-aie4'
$env:FLM_AIE4_CORELIB_PATH=$core
$env:FLM_CONFIG_PATH=Join-Path $root 'src/model_list.json'
$env:FLM_XCLBIN_PATH=Join-Path $root 'src'
$runtime=Split-Path $core
$env:PATH="$(Join-Path $root 'src/lib/xrt');$(Join-Path $root 'src/lib');$runtime;C:/Users/chiz/.conda/envs/hybrid-llm/Library/bin;C:/Users/chiz/work/hybrid-llm/install/xrt_package/xrt;$env:PATH"
New-Item -ItemType Directory -Force $outDir | Out-Null
$record=[ordered]@{started=(Get-Date).ToString('o');passed=$false;commands=@();host=[ordered]@{};provenance=[ordered]@{};files=@();cli=[ordered]@{};rest=[ordered]@{};performance=[ordered]@{};failures=@()}
# Progress markers go to stdout so a run that stalls can be located from the
# transcript alone; a silent 30-minute stall is indistinguishable from work.
function Mark([string]$m){Write-Host ("[mark] "+(Get-Date).ToString('HH:mm:ss.fff')+" "+$m)}
# ConvertTo-Json cannot be used on the record as a whole. Some of the values it
# holds are live .NET objects whose property graphs loop back on themselves, and
# ConvertTo-Json expands such a graph until -Depth runs out, which allocates tens
# of gigabytes and never returns. This was confirmed on both Windows PowerShell
# 5.1 and PowerShell 7.0.0 (3.4 GB and still climbing when killed) — moving to a
# newer engine does not avoid it, so do not remove this. This emitter walks
# the record itself: it refuses to descend past $script:JsonMaxDepth, and it
# refuses to re-enter an object that is already an ancestor of the current node.
# ConvertTo-Json is still used, but only ever on a single scalar string.
$script:JsonMaxDepth=10
function JsonScalar($s){return (ConvertTo-Json -InputObject ([string]$s))}
function EmitJson($v,[string]$label,[int]$level,$ancestors){
  if($null -eq $v){return 'null'}
  if($v -is [string]){return (JsonScalar $v)}
  if($v -is [bool]){if($v){return 'true'}else{return 'false'}}
  if($v -is [datetime]){return (JsonScalar $v.ToString('o'))}
  if($v -is [double] -or $v -is [single]){if([double]::IsNaN($v)-or[double]::IsInfinity($v)){return 'null'};return (([double]$v).ToString('R',[Globalization.CultureInfo]::InvariantCulture))}
  if($v -is [ValueType] -and $v -isnot [char] -and $v -isnot [Enum]){return (([string]$v))}
  if($level -ge $script:JsonMaxDepth){return (JsonScalar $v)}
  # The ancestor test exists for live .NET objects, whose property graphs loop.
  # It deliberately does not apply to a PSCustomObject: ConvertFrom-Json only
  # ever builds trees, and every object it produces shares one singleton base
  # instance, so testing those would report every nested JSON object as a loop.
  $bo=$null;try{$bo=$v.PSObject.BaseObject}catch{}
  $next=$ancestors
  if($null -ne $bo -and $bo -isnot [System.Management.Automation.PSCustomObject]){
    foreach($a in $ancestors){if([object]::ReferenceEquals($a,$bo)){return (JsonScalar '<cycle>')}}
    # The ancestor list must be built with Add, not with "+". Adding an array
    # with "+" splices its elements in, which would put every element of an
    # array on the ancestor list and make each of them look like a loop.
    $next=New-Object Collections.ArrayList
    if($null -ne $ancestors){[void]$next.AddRange($ancestors)}
    [void]$next.Add($bo)
  }
  $parts=New-Object Collections.ArrayList
  if($v -is [System.Collections.IDictionary]){
    foreach($k in @($v.Keys)){
      $sw=[Diagnostics.Stopwatch]::StartNew()
      [void]$parts.Add((JsonScalar $k)+':'+(EmitJson $v[$k] "$label.$k" ($level+1) $next))
      if($level -lt 2){Mark ("json {0}.{1} in {2:N1}s" -f $label,$k,$sw.Elapsed.TotalSeconds)}
    }
    return '{'+($parts -join ',')+'}'
  }
  if($v -is [System.Collections.IEnumerable]){
    foreach($e in $v){[void]$parts.Add((EmitJson $e "$label[]" ($level+1) $next))}
    return '['+($parts -join ',')+']'
  }
  $props=@($v.PSObject.Properties)
  if($props.Count -gt 0){
    foreach($p in $props){
      $pv=$null;try{$pv=$p.Value}catch{$pv="<unreadable: $($_.Exception.Message)>"}
      [void]$parts.Add((JsonScalar $p.Name)+':'+(EmitJson $pv "$label.$($p.Name)" ($level+1) $next))
    }
    return '{'+($parts -join ',')+'}'
  }
  return (JsonScalar $v)
}
function WriteRecord($rec,[string]$path){
  [IO.File]::WriteAllText($path,"{`r`n",[Text.Encoding]::UTF8)
  $first=$true
  foreach($k in @($rec.Keys)){
    $sw=[Diagnostics.Stopwatch]::StartNew()
    try{$t=EmitJson $rec[$k] $k 1 (New-Object Collections.ArrayList)}catch{$t=JsonScalar ("<unserializable: "+$_.Exception.Message+">")}
    if(-not$first){[IO.File]::AppendAllText($path,",`r`n",[Text.Encoding]::UTF8)}
    $first=$false
    [IO.File]::AppendAllText($path,('  "{0}": {1}' -f $k,$t),[Text.Encoding]::UTF8)
    Mark ("json section {0} written in {1:N1}s" -f $k,$sw.Elapsed.TotalSeconds)
  }
  [IO.File]::AppendAllText($path,"`r`n}`r`n",[Text.Encoding]::UTF8)
}
# Piping an ErrorRecord to Out-String yields nothing but a newline under some
# host configurations, which would record a failure with no reason attached.
function ErrText($e){
  $parts=@("$($e.Exception.GetType().FullName): $($e.Exception.Message)")
  $rendered=($e|Out-String);if(-not [string]::IsNullOrWhiteSpace($rendered)){$parts+=$rendered.Trim()}
  if($e.InvocationInfo -and $e.InvocationInfo.PositionMessage){$parts+=$e.InvocationInfo.PositionMessage.Trim()}
  if($e.ScriptStackTrace){$parts+=$e.ScriptStackTrace.Trim()}
  return ($parts -join "`n")
}
function Cmd([string]$line,[scriptblock]$body){$start=Get-Date;try{&$body;$ec=$LASTEXITCODE;if($null-eq$ec){$ec=0}}catch{$ec=1;$t=ErrText $_;$record.failures+=$t;Mark ("FAILURE in ${line}: "+$t);throw}finally{$record.commands+=@([ordered]@{command=$line;exit_code=$ec;seconds=((Get-Date)-$start).TotalSeconds})}}
# A non-2xx reply is an error in both engines, but the two expose it
# differently: Windows PowerShell hands back a WebResponse to read a stream
# from, PowerShell 7 hands back an HttpResponseMessage and puts the body in
# ErrorDetails. An expected 400 must not depend on which engine is running.
function Post([string]$path,$body,[int]$TimeoutSec=900){
  try{$r=Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port$path" -Method Post -ContentType 'application/json' -TimeoutSec $TimeoutSec -Body ($body|ConvertTo-Json -Depth 8 -Compress);return [ordered]@{status=[int]$r.StatusCode;text=$r.Content;json=($r.Content|ConvertFrom-Json)}}
  catch{
    $resp=$_.Exception.Response
    if($null -eq $resp){throw}
    $text=$null
    if($_.ErrorDetails -and $_.ErrorDetails.Message){$text=$_.ErrorDetails.Message}
    elseif($resp.PSObject.Methods['GetResponseStream']){$text=(New-Object IO.StreamReader($resp.GetResponseStream())).ReadToEnd()}
    elseif($resp.Content){$text=$resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()}
    $json=$null;if(-not [string]::IsNullOrWhiteSpace($text)){try{$json=$text|ConvertFrom-Json}catch{}}
    return [ordered]@{status=[int]$resp.StatusCode;text=$text;json=$json}
  }
}
# Two things must never reach curl as inline arguments. A JSON body loses its
# double quotes to PowerShell's native-argument quoting and the server sees a
# malformed object, so every body goes to a file and is read back with "@file".
# A header value containing a space is split into two arguments, so the second
# half is taken as another URL ("Could not resolve host: application"); the
# colon form without a space carries the same meaning and cannot split.
$ContentTypeArg='Content-Type:application/json'
function BodyFile([string]$name,$body){$p=Join-Path $outDir $name;Set-Content -Path $p -Value ($body|ConvertTo-Json -Depth 8 -Compress) -Encoding ASCII -NoNewline;return $p}
function CurlStream([string]$name,[string]$path,$body){$f=BodyFile $name $body;$out=(&curl.exe -sS -N -H $ContentTypeArg -d "@$f" "http://127.0.0.1:$Port$path" 2>&1|Out-String);if($LASTEXITCODE-ne 0){throw "curl failed ($LASTEXITCODE) for ${path}: $out"};if($out-match 'Could not resolve host'){throw "curl argument splitting for ${path}: $out"};return $out}
function CurlBackground([string]$name,[string]$path,$body,[string]$outFile){$f=BodyFile $name $body;return (Start-Process curl.exe -ArgumentList @('-sS','-N','-H',$ContentTypeArg,'-d',"@$f","http://127.0.0.1:$Port$path") -RedirectStandardOutput $outFile -PassThru)}
try{
 $record.host.computer=$env:COMPUTERNAME;$record.host.cpu=(Get-CimInstance Win32_Processor).Name;$record.host.npu=(Get-CimInstance Win32_PnPEntity|Where-Object Name -match 'NPU|Neural').Name;$os=Get-CimInstance Win32_OperatingSystem;$record.host.windows="$($os.Caption) $($os.Version) build $($os.BuildNumber)";$record.host.power=(powercfg /getactivescheme|Out-String).Trim()
 $record.provenance.fastflow=(git -C $root rev-parse HEAD).Trim();$coreRoot=(Resolve-Path (Join-Path $runtime '..')).Path;$record.provenance.corelib=(git -C $coreRoot rev-parse HEAD).Trim();$record.provenance.corelib_abi='0.3.0';$record.provenance.gguf_revision='78eb92a46fc37e6b524df991ed9aca9bc6aa7b80';$record.provenance.tokenizer_revision='cfbefacb99257ffa30c83adab238a50856ac3083';$record.provenance.corelib_sha256=(Get-FileHash $core -Algorithm SHA256).Hash.ToLower()
 Cmd "$exe check $Model" {&$exe check $Model|Out-Host;if($LASTEXITCODE-ne 0){throw 'check failed'}}
 $names=@('Phi-4-mini-instruct.Q8_0.gguf','tokenizer.json','tokenizer_config.json','config.json');$actual=@(Get-ChildItem $modelDir -File|% Name);if((Compare-Object ($names|Sort-Object) ($actual|Sort-Object))){throw 'model directory is not exactly four files'};foreach($n in $names){$f=Get-Item (Join-Path $modelDir $n);$record.files+=@([ordered]@{name=$n;bytes=$f.Length;sha256=(Get-FileHash $f.FullName -Algorithm SHA256).Hash.ToLower()})}
 $py=@'
from winpty import PtyProcess
import os,sys,time,threading,json,re
exe,model,out=sys.argv[1:4]
def run(cmds,timeout=900):
 p=PtyProcess.spawn(f'{exe} run {model}',env=os.environ.copy(),dimensions=(50,200));chunks=[]
 def rd():
  while p.isalive():
   try: chunks.append(p.read(8192))
   except: break
 threading.Thread(target=rd,daemon=True).start(); end=time.time()+timeout
 while time.time()<end and 'Type /? for help' not in ''.join(chunks) and p.isalive():time.sleep(.05)
 if not p.isalive():raise RuntimeError('CLI exited before prompt')
 def send(line):
  for ch in line:
   p.write(ch); time.sleep(.025)
  p.write('\r'); time.sleep(.5)
 for c in cmds:
  send(c)
  if c == '/bye': break
  time.sleep(1 if c.startswith('/') else 15)
 if cmds[-1]!='/bye': send('/bye')
 limit=time.time()+120
 while p.isalive() and time.time()<limit:time.sleep(.05)
 if p.isalive():p.terminate(force=True);raise RuntimeError('CLI did not self-terminate')
 text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','', ''.join(chunks))
 return {'exit_code':p.exitstatus,'text':text}
prompts=['What is 2+2? Answer with only the number.','What does AMD do? Answer in one sentence.','Name one primary color.','What is the capital of France?','Give one use of a GPU.','What is 3 times 3?','Name one programming language.','What does CPU stand for?','Say hello.','What is water made of?']
one=run(['/set gen-lim 32']+prompts+['/history','/status','/bye'])
cycles=[]
for i,q in enumerate(prompts):cycles.append({'prompt':q,**run(['/set gen-lim 8',q,'/history','/status','/bye'])})
json.dump({'prompts':prompts,'one_process':one,'cycles':cycles},open(out,'w',encoding='utf8'),indent=2)
'@
 $pyPath=Join-Path $outDir 'accept_cli.py';$cliPath=Join-Path $outDir 'accept_cli.json';Set-Content -Path $pyPath -Value $py -Encoding UTF8
 Mark 'starting CLI matrix'
 if($SkipCli){$record.skipped_cli=$true;Mark 'CLI matrix SKIPPED (diagnostic run, cannot pass)'}else{
 Cmd 'python accept_cli.py (10 prompts + 10 cycles)' {&$Python $pyPath $exe $Model $cliPath;if($LASTEXITCODE-ne 0){throw 'CLI matrix failed'}};$record.cli=Get-Content -Raw $cliPath|ConvertFrom-Json
 Mark 'CLI matrix done'
 $allCli=$record.cli.one_process.text;if($allCli-notmatch '(?m)\b4\b'){throw 'CLI arithmetic answer missing 4'};if($allCli-notmatch '(?i)AMD|semiconductor|processor|comput'){throw 'CLI AMD answer irrelevant'};if($allCli-notmatch 'corelib_aie4_gguf' -or $allCli-notmatch [regex]::Escape($core)){throw 'CLI backend/DLL proof missing'};if($allCli-match 'nan|-nan\(ind\)'){throw 'CLI profile reported a nan speed'};foreach($c in $record.cli.cycles){if($c.exit_code-ne 0 -or $c.text-notmatch 'Tokens:\s*[1-9]'){throw "CLI cycle failed: $($c.prompt)"}}}
 $serverLog=Join-Path $outDir 'accept-server.log';$serverErr=Join-Path $outDir 'accept-server.err.log';$server=Start-Process $exe -ArgumentList @('serve',$Model,'--port',$Port) -WorkingDirectory $root -RedirectStandardOutput $serverLog -RedirectStandardError $serverErr -PassThru
 try{for($i=0;$i-lt 300;$i++){try{$v=Invoke-RestMethod "http://127.0.0.1:$Port/api/version";break}catch{Start-Sleep -Milliseconds 200}};if(-not$v){throw 'server not ready'}
  Mark 'server ready'
  $apiNon=Post '/api/chat' @{model=$Model;messages=@(@{role='user';content='What is 2+2?'});stream=$false;options=@{num_predict=16}}
  Mark 'api_chat_nonstream done'
  $oaNon=Post '/v1/chat/completions' @{model=$Model;messages=@(@{role='user';content='What does AMD do?'});stream=$false;max_tokens=24}
  Mark 'openai_nonstream done'
  $apiStream=CurlStream 'body-api-stream.json' '/api/chat' @{model=$Model;messages=@(@{role='user';content='Say hello.'});stream=$true;options=@{num_predict=8}}
  Mark 'api_chat_stream done'
  $oaStream=CurlStream 'body-openai-stream.json' '/v1/chat/completions' @{model=$Model;messages=@(@{role='user';content='Name one GPU use.'});stream=$true;max_tokens=12}
  Mark 'openai_stream done'
  if($apiNon.status-ne 200-or$oaNon.status-ne 200-or[string]::IsNullOrWhiteSpace($apiStream)-or[string]::IsNullOrWhiteSpace($oaStream)){throw 'REST API matrix failed'}
  foreach($s in @($apiStream,$oaStream)){if($s-match '"error"'){throw "streaming response returned an error: $s"}}
  $cancelOut=Join-Path $outDir 'cancel-stream.txt'
  $cp=CurlBackground 'body-cancel.json' '/api/chat' @{model=$Model;request_id='accept-cancel';messages=@(@{role='user';content='Count upward for a long time.'});stream=$true;options=@{num_predict=1024}} $cancelOut
  Start-Sleep -Milliseconds 1500;$cancel=Post '/api/cancel' @{request_id='accept-cancel'};if(-not$cp.WaitForExit(300000)){$cp.Kill();throw 'the cancelled stream did not end'}
  Mark 'cancellation done'
  $recovery=Post '/api/chat' @{model=$Model;messages=@(@{role='user';content='What is 2+2?'});stream=$false;options=@{num_predict=8}}
  if(-not$cancel.json.cancelled-or$recovery.status-ne 200){throw 'cancellation recovery failed'}
  Mark 'recovery done'
  $probe=Post '/api/chat' @{model=$Model;messages=@(@{role='user';content='x'});stream=$false;options=@{num_predict=1}};$pt=[int]$probe.json.prompt_eval_count;$remaining=4095-$pt
  Mark "probe done prompt_tokens=$pt remaining=$remaining"
  $bOut=Join-Path $outDir 'boundary-stream.txt'
  $bp=CurlBackground 'body-boundary.json' '/api/chat' @{model=$Model;request_id='boundary4095';messages=@(@{role='user';content='x'});stream=$true;options=@{num_predict=$remaining}} $bOut
  Start-Sleep -Milliseconds 1500;$bcancel=Post '/api/cancel' @{request_id='boundary4095'};if(-not$bp.WaitForExit(300000)){$bp.Kill();throw 'the cancelled 4095 stream did not end'}
  Mark 'boundary 4095 cancel done'
  $b4096=Post '/api/chat' @{model=$Model;messages=@(@{role='user';content='x'});stream=$false;options=@{num_predict=($remaining+1)}}
  Mark "boundary 4096 done status=$($b4096.status)"
  if(-not$bcancel.json.cancelled-or$b4096.status-ne 400){throw "boundary behavior failed: 4095 cancelled=$($bcancel.json.cancelled) 4096 status=$($b4096.status)"}
  $record.rest=[ordered]@{api_chat_nonstream=$apiNon;openai_nonstream=$oaNon;api_chat_stream=$apiStream;openai_stream=$oaStream;cancellation=$cancel;recovery=$recovery;prompt_tokens=$pt;boundary4095_cancel=$bcancel;boundary4096=$b4096}
  $decodeTps=$null;if([double]$apiNon.json.eval_duration -gt 0){$decodeTps=[double]$apiNon.json.eval_count*1e9/[double]$apiNon.json.eval_duration}
  if($null-eq$decodeTps){throw 'decode duration was not reported; decode throughput cannot be recorded'}
  $record.performance=[ordered]@{load_ns=$apiNon.json.load_duration;cold_ttft_ns=$apiNon.json.prompt_eval_duration;warm_ttft_ns=$probe.json.prompt_eval_duration;decode_tokens=$apiNon.json.eval_count;decode_duration_ns=$apiNon.json.eval_duration;decode_tokens_per_second=$decodeTps}
 }finally{if($server-and-not$server.HasExited){Stop-Process -Id $server.Id -Force};Mark 'server stopped';$record.rest.server_log=(Get-Content -Raw $serverLog -ErrorAction SilentlyContinue)}
 if($record.rest.server_log-match '(?i)CPU fallback|phi4_npu|Q4NX'){throw 'fallback backend appeared in server log'}
 if($SkipCli){throw 'diagnostic -SkipCli run: the CLI matrix was not executed, so this record cannot report success'}
 $record.passed=$true
}catch{$t=ErrText $_;$record.failures+=$t;Mark ('FAILURE: '+$t);throw}finally{$record.finished=(Get-Date).ToString('o');Mark 'writing record';WriteRecord $record $outPath;Mark 'record written'}
