async function cariHukum() {
    const query = document.getElementById('queryInput').value;
    if (!query) return alert('Silakan masukkan pertanyaan hukum Anda!');

    document.getElementById('loadingText').style.display = 'block';
    document.getElementById('resultBox').style.display = 'none';
    document.getElementById('sourceList').innerHTML = '';

    try {
        const response = await fetch('/tanya', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ kueri: query })
        });
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        document.getElementById('aiAnswer').innerHTML = marked.parse(data.jawaban);
        
        data.sumber.forEach(doc => {
            document.getElementById('sourceList').innerHTML += `
                <div class="source-item">
                    <span class="source-pasal">
                        ${doc.pasal} 
                        <span class="source-skor">Akurasi: ${(doc.skor * 100).toFixed(1)}%</span>
                    </span>
                    <div style="font-size: 14px; color: #555;">${doc.teks_isi}</div>
                </div>
            `;
        });

        document.getElementById('resultBox').style.display = 'block';
    } catch (error) {
        alert('Terjadi kesalahan sistem: ' + error.message);
    } finally {
        document.getElementById('loadingText').style.display = 'none';
    }
}