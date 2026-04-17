// Ether Engine Main UI Logic

document.addEventListener('DOMContentLoaded', () => {

    // 1. Sidebar Navigation & Tab Switching
    const navItems = document.querySelectorAll('.sidebar-nav li');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const headerContext = document.getElementById('headerContext');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            // Update Active State
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');

            // Hide all tabs
            tabPanes.forEach(pane => pane.classList.remove('active'));

            // Show selected tab
            const target = item.getAttribute('data-target');
            const targetPane = document.getElementById(`tab-${target}`);
            if (targetPane) {
                targetPane.classList.add('active');
            }

            // Update Breadcrumb Header
            if (headerContext) {
                headerContext.textContent = target.toUpperCase();
            }
        });
    });

    // 2. Tab Pills Toggle Globally
    const tabPillsContainers = document.querySelectorAll('.tab-pills');
    tabPillsContainers.forEach(container => {
        const pills = container.querySelectorAll('.pill');
        pills.forEach(pill => {
            pill.addEventListener('click', () => {
                pills.forEach(p => p.classList.remove('active'));
                pill.classList.add('active');
            });
        });
    });
    
    // Toggle switches 
    const toggles = document.querySelectorAll('.toggle');
    toggles.forEach(toggle => {
       toggle.addEventListener('click', () => {
          toggle.classList.toggle('active');
       });
    });

    // 3. Dropzone Simulation (Dashboard & Upload tab)
    const dropzones = document.querySelectorAll('.sync-dropzone');
    
    dropzones.forEach(dropzone => {
       dropzone.addEventListener('dragover', (e) => {
           e.preventDefault();
           dropzone.style.borderColor = 'var(--primary)';
           dropzone.style.background = 'linear-gradient(180deg, rgba(0, 229, 255, 0.05) 0%, rgba(0, 229, 255, 0.1) 100%)';
       });

       dropzone.addEventListener('dragleave', () => {
           dropzone.style.borderColor = 'rgba(0, 229, 255, 0.3)';
           dropzone.style.background = 'linear-gradient(180deg, rgba(0, 229, 255, 0.01) 0%, rgba(0, 229, 255, 0.04) 100%)';
       });

       dropzone.addEventListener('drop', (e) => {
           e.preventDefault();
           
           // If it's the dashboard dropzone with progress bar
           const syncStatus = dropzone.querySelector('#syncStatus');
           const syncPercent = dropzone.querySelector('#syncPercent');
           const syncBar = dropzone.querySelector('#syncBar');

           if(syncStatus && syncPercent && syncBar) {
              syncStatus.textContent = "UPLOADING...";
              let progress = 0;
              
              const interval = setInterval(() => {
                  progress += Math.floor(Math.random() * 15);
                  if (progress >= 100) {
                      progress = 100;
                      clearInterval(interval);
                      syncStatus.textContent = "NEURAL MAPPING COMPLETE";
                      syncStatus.style.color = 'var(--accent-green)';
                      syncBar.style.background = 'var(--accent-green)';
                      syncBar.style.boxShadow = '0 0 10px var(--accent-green)';
                      
                      const tbody = document.getElementById('streamsTableBody');
                      if(tbody) {
                         const row = document.createElement('tr');
                         row.innerHTML = `
                             <td class="text-teal">DS-NEW-UPLOAD</td>
                             <td>0.084</td>
                             <td>-- Cols</td>
                             <td><span class="status-badge live">PROCESSING</span></td>
                             <td><i class="fa-solid fa-ellipsis-vertical" style="cursor: pointer; color: var(--text-muted)"></i></td>
                         `;
                         tbody.insertBefore(row, tbody.firstChild);
                      }
                  }
                  syncPercent.textContent = progress + "%";
                  syncBar.style.width = progress + "%";
              }, 300);
           } else {
              // Just visual feedback for big dropzone
              dropzone.style.borderColor = 'var(--accent-green)';
              dropzone.innerHTML = `<h3 style='color:var(--accent-green)'><i class='fa-solid fa-check'></i> Protocol Initiated</h3>`;
           }
       });
    });

    // 4. Live Data Simulation (Total Rows slowly increasing)
    const valTotalRows = document.getElementById('valTotalRows');
    let baseRows = 1280000;
    
    setInterval(() => {
        if (Math.random() > 0.5 && valTotalRows) {
            baseRows += Math.floor(Math.random() * 50);
            const formatted = (baseRows / 1000000).toFixed(2);
            valTotalRows.innerHTML = `${formatted}M <span class="badge positive">+12.4%</span>`;
        }
    }, 2000);
    
});
