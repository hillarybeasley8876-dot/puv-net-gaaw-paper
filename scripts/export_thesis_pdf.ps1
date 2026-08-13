param(
    [Parameter(Mandatory = $true)][string]$DocxPath,
    [Parameter(Mandatory = $true)][string]$PdfPath
)

$docx = [System.IO.Path]::GetFullPath($DocxPath)
$pdf = [System.IO.Path]::GetFullPath($PdfPath)
if (-not (Test-Path -LiteralPath $docx)) {
    throw "DOCX not found: $docx"
}
$pdfDir = [System.IO.Path]::GetDirectoryName($pdf)
if (-not (Test-Path -LiteralPath $pdfDir)) {
    New-Item -ItemType Directory -Path $pdfDir | Out-Null
}

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($docx, $false, $false)
    $doc.Repaginate()

    foreach ($story in $doc.StoryRanges) {
        $range = $story
        while ($null -ne $range) {
            if ($range.Fields.Count -gt 0) { $range.Fields.Update() | Out-Null }
            $range = $range.NextStoryRange
        }
    }
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Repaginate()
    $doc.Save()

    # 17 = wdExportFormatPDF; 0 = document content; 0 = print optimization.
    $doc.ExportAsFixedFormat($pdf, 17, $false, 0, 0)
    Write-Output "Exported $pdf"
}
finally {
    if ($null -ne $doc) { $doc.Close($false) }
    if ($null -ne $word) { $word.Quit() }
    if ($null -ne $doc) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    if ($null -ne $word) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
