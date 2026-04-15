document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('tab--active'));
    tab.classList.add('tab--active');

    const tabName = (tab as HTMLElement).dataset.tab;
    document.querySelectorAll('.tab-panel').forEach((panel) => {
      (panel as HTMLElement).style.display = 'none';
    });

    const activePanel = document.getElementById(`${tabName}-tab`);
    if (activePanel) activePanel.style.display = 'block';
  });
});
