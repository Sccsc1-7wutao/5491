# Keeps local evaluation processes at Idle CPU priority and Very-Low I/O priority so the desktop stays usable.
# Checks every 20 s; stops when the file io_nice_watcher.stop exists next to this script.
Add-Type -TypeDefinition @'
using System; using System.Runtime.InteropServices;
public static class IoPrio2 {
  [DllImport("ntdll.dll")] public static extern int NtSetInformationProcess(IntPtr h, int cls, ref int info, int len);
  [DllImport("kernel32.dll")] public static extern IntPtr OpenProcess(int acc, bool inh, int pid);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
  public static int Set(int pid, int prio) { IntPtr h = OpenProcess(0x0200|0x1000, false, pid); if (h==IntPtr.Zero) return -1; int v=prio; int r=NtSetInformationProcess(h, 33, ref v, 4); CloseHandle(h); return r; }
}
'@
$stop = Join-Path $PSScriptRoot 'io_nice_watcher.stop'
$seen = @{}
while (-not (Test-Path $stop)) {
  $procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(python|vot|cmd)\.exe$' -and $_.CommandLine -match 'test_rgbe_mgpus|test_rgbt_mgpus|run_offline_depthtrack|vot analysis|run_local_queue|lasher_eval|visevent_eval|fed_state_to_ckpt' }
  $ids = @($procs | ForEach-Object { $_.ProcessId })
  $kids = Get-CimInstance Win32_Process | Where-Object { $ids -contains $_.ParentProcessId } | ForEach-Object { $_.ProcessId }
  foreach ($id in ($ids + $kids | Select-Object -Unique)) {
    if (-not $seen.ContainsKey($id)) {
      try { (Get-Process -Id $id -ErrorAction Stop).PriorityClass = 'Idle' } catch {}
      [void][IoPrio2]::Set($id, 0)
      $seen[$id] = 1
    }
  }
  Start-Sleep -Seconds 20
}
