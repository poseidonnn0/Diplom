document.addEventListener('DOMContentLoaded', function() {
    let isCameraUpdating = false;

    function refreshVideoFeed() {
        const liveFeed = document.getElementById('liveFeed');
        const overlay = document.getElementById('videoOverlay');
        
        if (!liveFeed) return;

        liveFeed.onerror = () => {
            overlay.textContent = 'Ошибка подключения к камере';
            overlay.style.display = 'flex';
            setTimeout(() => {
                liveFeed.src = "/video_feed?" + Date.now();
            }, 3000);
        };

        liveFeed.onload = () => overlay.style.display = 'none';
        liveFeed.src = "/video_feed?" + Date.now();
    }

    setInterval(refreshVideoFeed, 3000);
    refreshVideoFeed();

    document.getElementById('save-camera-btn')?.addEventListener('click', async function(e) {
        e.preventDefault();
        if (isCameraUpdating) return;
        isCameraUpdating = true;
        
        const type = document.getElementById('camera-type').value;
        const url = document.getElementById('camera-url').value;

        try {
            const response = await fetch('/admin/update_camera', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ type, url })
            });

            const result = await response.json();
            if (!response.ok) throw new Error(result.message);
            
            refreshVideoFeed();
            alert('Настройки камеры обновлены!');
        } catch (err) {
            console.error('Ошибка:', err);
            alert('Ошибка: ' + err.message);
        } finally {
            isCameraUpdating = false;
        }
    });

    document.querySelectorAll('.save-zone-btn').forEach(btn => {
        btn.addEventListener('click', async function(e) {
            e.preventDefault();
            const index = parseInt(btn.dataset.zoneIndex);
            const card = document.querySelectorAll('.zone-card')[index];
            
            const data = {
                index: index,
                name: card.querySelector('.zone-name').value,
                color: card.querySelector('.zone-color').value,
                max_capacity: parseInt(card.querySelector('.zone-limit').value),
                coords: Array.from(card.querySelectorAll('.coords-grid input')).map(i => parseInt(i.value))
            };

            try {
                const response = await fetch('/admin/update_zone', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });

                const result = await response.json();
                if (!response.ok) throw new Error(result.message);
                
                alert('Зона сохранена!');
                location.reload();
            } catch (err) {
                console.error('Ошибка:', err);
                alert('Ошибка: ' + err.message);
            }
        });
    });
});