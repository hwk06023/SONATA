document.addEventListener('DOMContentLoaded', function() {
  // Initialize Mermaid diagrams if available
  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: true,
      theme: document.body.classList.contains('dark-mode') ? 'dark' : 'default',
      flowchart: { useMaxWidth: true, htmlLabels: true },
      securityLevel: 'loose'
    });
  }

  // Copy Code Button
  const codeBlocks = document.querySelectorAll('pre.highlight');
  codeBlocks.forEach(block => {
    const button = document.createElement('button');
    button.className = 'copy-code-button';
    button.innerHTML = '<i class="fas fa-copy"></i>';
    button.addEventListener('click', function() {
      const code = block.querySelector('code').innerText;
      navigator.clipboard.writeText(code);
      button.innerHTML = '<i class="fas fa-check"></i>';
      setTimeout(() => {
        button.innerHTML = '<i class="fas fa-copy"></i>';
      }, 2000);
    });
    block.appendChild(button);
  });

  // Initialize tabs
  const tabContainers = document.querySelectorAll('.tab-container');
  tabContainers.forEach(container => {
    const navItems = container.querySelectorAll('.tab-nav .tab-item');
    const contentPanes = container.querySelectorAll('.tab-content .tab-pane');
    
    navItems.forEach((item, index) => {
      item.addEventListener('click', () => {
        navItems.forEach(i => i.classList.remove('active'));
        contentPanes.forEach(p => p.classList.remove('active'));
        
        item.classList.add('active');
        contentPanes[index].classList.add('active');
      });
    });
    
    if (navItems.length > 0) {
      navItems[0].classList.add('active');
      contentPanes[0].classList.add('active');
    }
  });
  
  // Dark mode toggle
  const darkModeToggle = document.querySelector('.dark-mode-toggle');
  if (darkModeToggle) {
    darkModeToggle.addEventListener('click', () => {
      document.body.classList.toggle('dark-mode');
      document.body.classList.toggle('dark-mode-auto');
      
      if (window.mermaid) {
        window.mermaid.initialize({
          theme: document.body.classList.contains('dark-mode') ? 'dark' : 'default'
        });
      }
    });
  }
  
  // Enhanced search functionality
  const headerSearch = document.querySelector('.js-search');
  if (headerSearch) {
    const searchInput = headerSearch.querySelector('input');
    
    // Keyboard shortcut for search: / key
    document.addEventListener('keydown', function(e) {
      if (e.key === '/' && document.activeElement !== searchInput) {
        e.preventDefault();
        searchInput.focus();
      }
      
      // Escape key to close search
      if (e.key === 'Escape' && document.activeElement === searchInput) {
        searchInput.blur();
        searchInput.value = '';
      }
    });
  }
  
  // Make external links open in new tab
  document.querySelectorAll('a[href^="http"]').forEach(link => {
    if (!link.hasAttribute('target') && link.hostname !== window.location.hostname) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener');
    }
  });
  
  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const targetId = this.getAttribute('href');
      if (targetId !== '#' && targetId !== '#top') {
        const targetElement = document.querySelector(targetId);
        if (targetElement) {
          e.preventDefault();
          targetElement.scrollIntoView({
            behavior: 'smooth'
          });
          history.pushState(null, null, targetId);
        }
      }
    });
  });
}); 